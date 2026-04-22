"""
app.py — Streamlit entry point for the Robo-Adviser AI Chatbot.

Layout (single page):
  * Top header: title + stage progress + LLM status chip
  * Main body (2 columns, 60/40):
      Left  - chat area (st.chat_message / st.chat_input)
      Right - dashboard tabs: Radar / Frontier / Weights / Backtest
  * Footer: action buttons (Restart / Export PDF / What-if slider)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from config import FUND_MAP, LLM_CONFIG, QUESTIONS
from data_loader import fund_display_name
from engine import optimize_portfolio
from llm_client import set_llm_enabled
from state_machine import (
    BotSegment,
    Phase,
    SessionState,
    ensure_prices_loaded,
    greet,
    handle_user_input,
)
from visuals import (
    plot_backtest,
    plot_efficient_frontier,
    plot_risk_radar,
    plot_weights_pie,
)

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MDFinTech Robo-Adviser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --navy: #0B2545;
  --midnight: #13315C;
  --gold: #C9A227;
  --grey-soft: #F7F9FC;
  --grey-border: #EEF2F7;
  --text-secondary: #5A6B85;
}
html, body, [class*="css"] { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

/* Header bar */
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px; border-radius: 14px;
  background: linear-gradient(135deg, var(--navy) 0%, var(--midnight) 100%);
  color: white; margin-bottom: 18px;
  box-shadow: 0 4px 18px rgba(11,37,69,0.15);
}
.app-header .title { font-weight: 700; font-size: 20px; letter-spacing: .3px; }
.app-header .subtitle { font-size: 12px; color: #C9D6E4; margin-top: 2px; }

.stage-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 999px;
  background: rgba(201,162,39,0.15); color: var(--gold);
  font-size: 12px; font-weight: 600;
}
.llm-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.08); color: white; font-size: 12px;
}
.llm-dot { width: 8px; height: 8px; border-radius: 50%; }
.llm-dot.on  { background: #2E8B57; box-shadow: 0 0 6px #2E8B57; }
.llm-dot.off { background: #8E9AAB; }

/* Chat bubbles tweak */
[data-testid="stChatMessage"] { padding: 10px 14px; }

/* Warning bubble */
.warn-card {
  border-left: 4px solid var(--gold);
  background: #FFFBEE; padding: 12px 14px; border-radius: 8px;
  color: #5B4A12; font-size: 14px;
}
.profile-card, .portfolio-card {
  background: white; border: 1px solid var(--grey-border); border-radius: 14px;
  padding: 14px 18px; box-shadow: 0 2px 8px rgba(11,37,69,0.04);
}
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0; }
.metric {
  background: var(--grey-soft); border-radius: 10px; padding: 10px 12px;
  border: 1px solid var(--grey-border);
}
.metric .label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .5px; }
.metric .value { font-size: 18px; font-weight: 700; color: var(--navy); margin-top: 2px; }
.metric .value.pos { color: #2E8B57; }
.metric .value.neg { color: #C0392B; }

/* Dashboard tabs */
div[data-testid="stTabs"] button { font-weight: 600; }

/* Fund-universe mini panel */
.fund-pill {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: #EEF2F7; color: var(--midnight); margin: 2px 4px 2px 0;
  font-size: 11px; font-weight: 600;
}

/* Footer buttons */
.footer-row { margin-top: 14px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "fsm" not in st.session_state:
        st.session_state.fsm = SessionState()
        st.session_state.chat_history: list[tuple[str, BotSegment | str]] = []
        # LLM runtime toggle — defaults to ON when a key is configured.
        st.session_state.llm_enabled = bool(LLM_CONFIG.available)
        for seg in greet():
            st.session_state.chat_history.append(("assistant", seg))

_init_session()

# Push session toggle down into the llm_client module every rerun.
set_llm_enabled(st.session_state.get("llm_enabled", False))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _stage_label(phase: Phase, current_q: int) -> str:
    if phase == Phase.WELCOME:        return "Welcome"
    if phase == Phase.QUESTIONNAIRE:  return f"Questionnaire {current_q}/10"
    if phase == Phase.CONFIRM_ANSWER: return f"Confirming Q{current_q}"
    if phase == Phase.CONFLICT_CHECK: return "Clarifying answers"
    if phase == Phase.PROFILE:        return "Profile ready"
    if phase == Phase.RECOMMEND:      return "Portfolio ready"
    if phase == Phase.EXPORT:         return "Exporting"
    return phase.value


def render_header() -> None:
    """Pure-HTML navy banner (no Streamlit widgets inside — avoids the big
    white padding block caused by st.columns in the header row).

    The LLM runtime toggle lives in the sidebar (see render_sidebar_controls)
    where it has a stable DOM position and doesn't fight the banner layout.
    """
    fsm: SessionState = st.session_state.fsm
    stage = _stage_label(fsm.phase, fsm.current_q)
    key_configured = bool(LLM_CONFIG.available)
    llm_on = bool(st.session_state.get("llm_enabled", False)) and key_configured
    llm_label = f"LLM: {LLM_CONFIG.model}" if llm_on else "LLM: Mock (offline)"
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="title">📊 MDFinTech Robo-Adviser</div>
            <div class="subtitle">BMD5302 · Part 3 · Markowitz + AI Chatbot</div>
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <span class="stage-chip">● {stage}</span>
            <span class="llm-chip"><span class="llm-dot {'on' if llm_on else 'off'}"></span>{llm_label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls() -> None:
    """LLM toggle + quick help live in the sidebar so the banner stays clean."""
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        key_configured = bool(LLM_CONFIG.available)
        if key_configured:
            st.toggle(
                "Enable LLM",
                key="llm_enabled",
                help=(
                    "ON: DeepSeek/OpenAI-compatible model powers "
                    "natural-language parsing and open-ended Q&A.\n\n"
                    "OFF: rule-based Mock only — offline-safe, deterministic."
                ),
            )
            on = bool(st.session_state.get("llm_enabled", False))
            st.caption(
                f"Model: `{LLM_CONFIG.model}`  \n"
                f"Status: {'🟢 live' if on else '🟡 muted (Mock mode)'}"
            )
        else:
            st.toggle("Enable LLM", value=False, disabled=True)
            st.caption("🔒 No API key configured. Add `OPENAI_API_KEY` in Secrets to enable.")

        st.divider()
        st.markdown("### 💬 Quick commands")
        st.caption(
            "- `start` — begin the questionnaire\n"
            "- `what if A = 2` — re-optimize with a new risk aversion\n"
            "- `explain sharpe` — metric deep-dive\n"
            "- `export pdf` — download advice report\n"
            "- `restart` — retake the questionnaire"
        )


render_sidebar_controls()
render_header()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col_chat, col_dash = st.columns([0.58, 0.42], gap="large")

# ----- Dashboard (right column) -----
with col_dash:
    tabs = st.tabs(["🎯 Risk radar", "📈 Frontier", "🥧 Weights", "📉 Backtest"])
    fsm: SessionState = st.session_state.fsm

    with tabs[0]:
        if fsm.scores:
            subscores = {
                next(q["category"] for q in QUESTIONS if q["id"] == qid): s
                for qid, s in fsm.scores.items()
            }
            st.plotly_chart(plot_risk_radar(subscores), use_container_width=True)
        else:
            st.info("Answer the questionnaire on the left to build your risk radar.")

    with tabs[1]:
        try:
            _, mu, sigma, src = ensure_prices_loaded()
            user_pt = None
            if fsm.metrics:
                user_pt = (fsm.metrics["std"], fsm.metrics["expected_return"])
            st.plotly_chart(
                plot_efficient_frontier(mu, sigma, user_point=user_pt),
                use_container_width=True,
            )
            st.caption(f"Data source: **{src}** · 10 funds · annualized moments.")
        except Exception as exc:
            st.warning(f"Couldn't build the frontier yet: {exc}")

    with tabs[2]:
        if fsm.weights:
            fund_names = {c: fund_display_name(c) for c in fsm.weights.keys()}
            st.plotly_chart(plot_weights_pie(fsm.weights, fund_names), use_container_width=True)

            # Weight table
            import pandas as pd
            df = pd.DataFrame([
                {"Fund": c, "Name": fund_display_name(c), "Weight": f"{w*100:.2f}%"}
                for c, w in sorted(fsm.weights.items(), key=lambda kv: -kv[1])
                if w >= 0.001
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Complete the questionnaire to see your allocation.")

    with tabs[3]:
        if fsm.weights:
            try:
                prices, *_ = ensure_prices_loaded()
                st.plotly_chart(plot_backtest(prices, fsm.weights), use_container_width=True)
            except Exception as exc:
                st.warning(f"Backtest unavailable: {exc}")
        else:
            st.info("A backtest will appear once you have a recommended portfolio.")


# ----- Chat (left column) -----
def _render_segment(seg: BotSegment) -> None:
    if seg.kind == "warning":
        st.markdown(f"<div class='warn-card'>{seg.text}</div>", unsafe_allow_html=True)
    elif seg.kind in ("profile", "portfolio"):
        cls = "profile-card" if seg.kind == "profile" else "portfolio-card"
        st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
        st.markdown(seg.text)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(seg.text)
    if seg.llm_source:
        st.caption(f"_generated via: {seg.llm_source}_")


with col_chat:
    st.markdown("### 💬 Chat with your adviser")
    chat_container = st.container(height=560, border=True)
    with chat_container:
        for role, content in st.session_state.chat_history:
            with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🤖"):
                if isinstance(content, BotSegment):
                    _render_segment(content)
                else:
                    st.markdown(content)

    user_text = st.chat_input("Type a number, a letter, or a full sentence...")
    if user_text:
        fsm: SessionState = st.session_state.fsm
        st.session_state.chat_history.append(("user", user_text))
        bot_segs = handle_user_input(fsm, user_text)
        for seg in bot_segs:
            st.session_state.chat_history.append(("assistant", seg))
        st.rerun()


# ---------------------------------------------------------------------------
# Footer: action buttons + What-if slider + fund universe
# ---------------------------------------------------------------------------

st.divider()

footer_cols = st.columns([1.2, 1.2, 1.2, 2.4])

with footer_cols[0]:
    if st.button("🔄 Restart questionnaire", use_container_width=True):
        st.session_state.pop("fsm", None)
        st.session_state.pop("chat_history", None)
        _init_session()
        st.rerun()

with footer_cols[1]:
    fsm: SessionState = st.session_state.fsm
    can_export = bool(fsm.weights)
    if st.button("📄 Export PDF", use_container_width=True, disabled=not can_export):
        try:
            from exporter import build_pdf
            pdf_path = build_pdf(fsm)
            st.session_state["last_pdf"] = str(pdf_path)
            st.toast("PDF ready — download below.", icon="✅")
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")
    if st.session_state.get("last_pdf"):
        pdf_path = Path(st.session_state["last_pdf"])
        if pdf_path.exists():
            st.download_button(
                "⬇ Download PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )

with footer_cols[2]:
    if st.button("🔁 Refresh market data", use_container_width=True,
                 help="Force-pull latest prices from Yahoo Finance"):
        try:
            ensure_prices_loaded(force=True)
            st.toast("Market data refreshed.", icon="✅")
            st.rerun()
        except Exception as exc:
            st.error(f"Refresh failed: {exc}")

with footer_cols[3]:
    fsm = st.session_state.fsm
    if fsm.weights:
        st.markdown("**What-if: tweak your risk aversion A**")
        new_A = st.slider("A (higher = more risk-averse)", 0.5, 10.0,
                          value=float(fsm.A_value), step=0.5, key="whatif_A")
        if abs(new_A - fsm.A_value) > 1e-6:
            _, mu, sigma, _src = ensure_prices_loaded()
            result = optimize_portfolio(mu, sigma, A=new_A, allow_short=False)
            fsm.A_value = new_A
            fsm.weights = {f"Fund_{i+1:02d}": float(w) for i, w in enumerate(result.weights)}
            fsm.metrics = {
                "expected_return": result.expected_return,
                "variance": result.variance,
                "std": result.std,
                "utility": result.utility,
                "sharpe": result.sharpe,
            }
            st.session_state.chat_history.append((
                "assistant",
                BotSegment(
                    text=(f"_What-if applied_: A = **{new_A}** → "
                          f"r = **{result.expected_return*100:.2f}%**, "
                          f"sigma = **{result.std*100:.2f}%**, "
                          f"Sharpe = **{result.sharpe:.2f}**"),
                    kind="text",
                ),
            ))
            st.rerun()

# Fund universe preview
with st.expander("🎯 Fund universe (10 ETFs mapped to FSMOne selection)"):
    pills = " ".join(
        f"<span class='fund-pill'>{code} · {meta['yahoo']} · {meta['name']}</span>"
        for code, meta in FUND_MAP.items()
    )
    st.markdown(pills, unsafe_allow_html=True)

st.caption(
    f"_BMD5302 Robo-Adviser · Part 3 · rendered at {datetime.now():%Y-%m-%d %H:%M}_"
)
