"""
llm_client.py — OpenAI-compatible LLM adapter + rule-based Mock fallback.

The rest of the application calls four high-level functions — it never cares
whether a real LLM or the rule-based Mock is behind the curtain:

* ``parse_answer(question, user_text)``
* ``generate_profile(total_score, level, A, subscores)``
* ``explain_portfolio(weights, metrics, A, data_source)``
* ``classify_followup(user_text)``

If no API key is configured (``LLM_CONFIG.available == False``), or if a live
call fails / times out, we degrade silently to the Mock implementation so the
demo never crashes in front of the graders.

No user messages are logged — only metadata (latency, token counts, success
flags) — to avoid leaking PII if students use real personal info during demos.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from config import (
    LLM_CONFIG,
    QUESTIONS,
    get_question,
    score_option,
)
from prompts import (
    EXPLAIN_PORTFOLIO_PROMPT,
    FOLLOWUP_INTENT_PROMPT,
    FREEFORM_SYSTEM_PROMPT,
    FREEFORM_USER_PROMPT,
    PARSE_ANSWER_PROMPT,
    PROFILE_REPORT_PROMPT,
    SYSTEM_PROMPT,
    format_history_block,
    format_options_block,
    format_top_holdings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime toggle — lets the UI force "Mock mode" without unsetting the API key.
# Call set_llm_enabled(False) to treat the client as offline for the current
# Python process (or a Streamlit session, since _ENABLED lives in module state
# but the toggle is re-applied from st.session_state on every rerun).
# ---------------------------------------------------------------------------

_ENABLED: bool = True


def set_llm_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def is_llm_active() -> bool:
    """True only if a key is configured AND the runtime toggle is on."""
    return bool(_ENABLED and LLM_CONFIG.available)


@dataclass
class ParseResult:
    choice: str | None
    score: int
    confidence: float
    need_clarify: bool
    reason: str
    source: str  # "llm" | "mock"


# ---------------------------------------------------------------------------
# Low-level OpenAI-compatible client (lazy init to speed up cold start)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not is_llm_active():
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(
            api_key=LLM_CONFIG.api_key,
            base_url=LLM_CONFIG.base_url,
            timeout=LLM_CONFIG.timeout,
            max_retries=LLM_CONFIG.max_retries,
        )
        return _client
    except Exception as exc:  # pragma: no cover
        logger.warning("OpenAI client init failed: %s", exc)
        return None


def _chat(prompt: str, expect_json: bool = False, temperature: float = 0.3,
          system_prompt: str | None = None) -> str | None:
    """Call the LLM. Returns None on any failure so callers can fall back."""
    if not is_llm_active():
        return None
    client = _get_client()
    if client is None:
        return None

    started = time.time()
    try:
        kwargs = dict(
            model=LLM_CONFIG.model,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
        )
        if expect_json:
            # Supported by DeepSeek / OpenAI / Qwen compatible mode
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        logger.info("LLM call ok (%.2fs, model=%s)", time.time() - started, LLM_CONFIG.model)
        return content
    except Exception as exc:
        logger.warning("LLM call failed (%.2fs): %s", time.time() - started, exc)
        return None


def _safe_json(text: str) -> dict | None:
    """Parse JSON robustly — strip ``` fences, handle stray commas etc."""
    if not text:
        return None
    # Strip markdown fences if present
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # Try full parse first
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try to find the first top-level JSON object
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Public API — all returns include a ``source`` indicator so the UI can show it
# ---------------------------------------------------------------------------

def parse_answer(question_id: int, user_text: str) -> ParseResult:
    """Map the user's free-form answer to A/B/C/D/E.

    Tries the LLM first; on any failure falls back to a rule-based parser that
    looks for (1) a direct letter mention, (2) keyword matching against option
    text. Confidence from the mock is capped at 0.6 so the UI can still decide
    to ask a confirmation.
    """
    user_text = (user_text or "").strip()
    q = get_question(question_id)
    if not user_text:
        return ParseResult(None, 0, 0.0, True, "empty answer", "mock")

    # --- LLM path ---
    prompt = PARSE_ANSWER_PROMPT.format(
        q_id=question_id,
        question_text=q["text"],
        options_block=format_options_block(q["options"]),
        user_text=user_text,
    )
    raw = _chat(prompt, expect_json=True, temperature=0.1)
    data = _safe_json(raw) if raw else None
    if data and data.get("choice") in {lbl for lbl, _, _ in q["options"]}:
        lbl = data["choice"]
        return ParseResult(
            choice=lbl,
            score=score_option(question_id, lbl),
            confidence=float(data.get("confidence", 0.8)),
            need_clarify=bool(data.get("need_clarify", False)),
            reason=str(data.get("reason", "")),
            source="llm",
        )

    # --- Mock fallback ---
    return _mock_parse_answer(question_id, user_text)


def _match_numeric_range(q: dict, user_text: str) -> str | None:
    """Heuristic for numeric questions (age, income, investment horizon, loss %).

    We look at the first number in the user's text and match it against option
    labels that express numeric ranges such as "18-30", "S$60,000 - 120,000",
    "Less than 2 years", "3 - 5 years", "Up to 10% loss".
    """
    import re as _re
    cat = q.get("category", "")
    numeric_cats = {
        "age", "income", "experience_years", "horizon",
        "loss_tolerance", "investable_ratio",
    }
    if cat not in numeric_cats:
        return None

    # Extract the first number (ignoring thousands commas).
    m = _re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", user_text.replace(",", ""))
    if not m:
        return None
    try:
        value = float(m.group(1))
    except Exception:
        return None

    # Income is typically given in thousands ('100k') or full SGD.
    if cat == "income":
        if "k" in user_text.lower() and value < 1000:
            value *= 1000
        if value < 60_000:            return "A"
        if value < 120_000:           return "B"
        if value < 180_000:           return "C"
        if value < 300_000:           return "D"
        return "E"

    if cat == "age":
        if value <= 30:               return "A"
        if value <= 50:               return "B"
        if value <= 60:               return "C"
        return "D"

    if cat == "experience_years":
        if value < 1:                 return "A"
        if value < 2:                 return "B"
        if value <= 5:                return "C"
        if value <= 8:                return "D"
        return "E"

    if cat == "horizon":
        if value < 1:                 return "A"
        if value <= 3:                return "B"
        if value <= 5:                return "C"
        return "D"

    if cat == "loss_tolerance":
        # pct assumed; if number has no % but <= 100 assume percent
        if value <= 0:                return "A"
        if value <= 5:                return "B"
        if value <= 10:               return "C"
        if value <= 50:               return "D"
        return "E"

    if cat == "investable_ratio":
        if value < 10:                return "A"
        if value < 25:                return "B"
        if value < 50:                return "C"
        return "D"

    return None


def _mock_parse_answer(question_id: int, user_text: str) -> ParseResult:
    q = get_question(question_id)
    lowered = user_text.lower().strip()

    # (1) Direct letter — "A", "选 B", "i'll go with c"
    m = re.search(r"(?:^|\b|选|pick|choose|answer)[\s:：]*([A-Ea-e])\b", user_text)
    if m:
        lbl = m.group(1).upper()
        if lbl in {l for l, _, _ in q["options"]}:
            return ParseResult(
                choice=lbl,
                score=score_option(question_id, lbl),
                confidence=0.55,
                need_clarify=False,
                reason="matched letter in user text",
                source="mock",
            )

    # (2) Numeric range heuristic (age, income, years, loss %)
    num_match = _match_numeric_range(q, user_text)
    if num_match is not None:
        return ParseResult(
            choice=num_match,
            score=score_option(question_id, num_match),
            confidence=0.6,
            need_clarify=False,
            reason="numeric range match",
            source="mock",
        )

    # (3) Keyword heuristics per-category
    import unicodedata
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).lower()
    lowered = _norm(user_text)

    # Score each option by counting how many of its lowercase tokens appear.
    best_lbl, best_hits = None, 0
    for lbl, text, _ in q["options"]:
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) >= 3]
        hits = sum(1 for t in tokens if t in lowered)
        if hits > best_hits:
            best_lbl, best_hits = lbl, hits

    if best_lbl and best_hits >= 1:
        return ParseResult(
            choice=best_lbl,
            score=score_option(question_id, best_lbl),
            confidence=0.45,
            need_clarify=True,
            reason=f"heuristic match (hits={best_hits})",
            source="mock",
        )

    return ParseResult(
        choice=None,
        score=0,
        confidence=0.0,
        need_clarify=True,
        reason="no confident match — please rephrase",
        source="mock",
    )


def generate_profile(
    total_score: int,
    level_code: str,
    level_name: str,
    A_value: float,
    subscores: dict[str, int],
) -> tuple[str, str]:
    """Generate the investor-persona report.  Returns (text, source)."""
    prompt = PROFILE_REPORT_PROMPT.format(
        total_score=total_score,
        level_code=level_code,
        level_name=level_name,
        A_value=A_value,
        subscores_json=json.dumps(subscores, ensure_ascii=False),
    )
    raw = _chat(prompt, expect_json=False, temperature=0.5)
    if raw and len(raw.strip()) > 40:
        return raw.strip(), "llm"
    return _mock_profile(total_score, level_code, level_name, A_value), "mock"


def _mock_profile(total_score: int, level_code: str, level_name: str, A_value: float) -> str:
    personas = {
        "R1": "a highly cautious investor who prizes capital preservation above all",
        "R2": "a steady, conservative investor who tolerates modest volatility for modest rewards",
        "R3": "a balanced investor happy to trade some stability for meaningful long-term growth",
        "R4": "a growth-seeking investor who accepts drawdowns in pursuit of compounding returns",
        "R5": "an aggressive investor willing to ride major swings in exchange for the highest potential upside",
    }
    desc = personas.get(level_code, "a balanced investor")
    return (
        f"**You are {desc}.** Your weighted questionnaire score is **{total_score}/75** "
        f"which places you in **{level_code} — {level_name}**. "
        f"You typically do well when markets trend steadily and you have time on your side; "
        f"the biggest pitfall for your profile is reacting emotionally to short-term drawdowns "
        f"and abandoning the plan at the worst possible moment. "
        f"\n\nYour recommended risk-aversion coefficient **A = {A_value}**."
    )


def explain_portfolio(
    weights: dict[str, float],
    metrics: dict[str, float],
    A_value: float,
    fund_names: dict[str, str],
    data_source: str,
) -> tuple[str, str]:
    prompt = EXPLAIN_PORTFOLIO_PROMPT.format(
        A_value=A_value,
        top_holdings_block=format_top_holdings(weights, fund_names),
        expected_return_pct=f"{metrics['expected_return'] * 100:.2f}%",
        std_pct=f"{metrics['std'] * 100:.2f}%",
        sharpe=metrics.get("sharpe", 0.0),
        utility=metrics.get("utility", 0.0),
        data_source=data_source,
    )
    raw = _chat(prompt, expect_json=False, temperature=0.5)
    if raw and len(raw.strip()) > 60:
        return raw.strip(), "llm"
    return _mock_explain(weights, metrics, A_value, fund_names, data_source), "mock"


def _mock_explain(
    weights: dict[str, float],
    metrics: dict[str, float],
    A_value: float,
    fund_names: dict[str, str],
    data_source: str,
) -> str:
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
    top = [(c, w) for c, w in top if w >= 0.01]
    lines: list[str] = []
    lines.append(
        f"- With a risk-aversion coefficient of **A = {A_value}**, the optimizer is "
        f"balancing expected return **{metrics['expected_return'] * 100:.2f}%** against "
        f"volatility **{metrics['std'] * 100:.2f}%** (Sharpe ≈ **{metrics.get('sharpe', 0):.2f}**)."
    )
    if top:
        names = ", ".join(f"**{c}** ({fund_names.get(c, c)}) at {w * 100:.1f}%" for c, w in top)
        lines.append(f"- Your top holdings are {names}.")
    lines.append(
        "- These weights sum to 100%, no short selling, which matches the constraints "
        "set in Part 1/2 of the project."
    )
    lines.append(
        "- In a strong month you should see gains broadly proportional to the headline return; "
        "in a weak month the volatility figure tells you roughly how large a drawdown to budget for."
    )
    lines.append(
        f"- _Caveat_: moments μ and Σ are estimated from {data_source} data — past performance "
        "does not guarantee future results. Re-run the optimizer as new prices arrive."
    )
    return "\n".join(lines)


def classify_followup(user_text: str) -> dict[str, Any]:
    """Classify a free-form follow-up message into one of a fixed set of intents."""
    prompt = FOLLOWUP_INTENT_PROMPT.format(user_text=user_text)
    raw = _chat(prompt, expect_json=True, temperature=0.1)
    data = _safe_json(raw) if raw else None
    if data and "intent" in data:
        return data | {"source": "llm"}

    # Mock classifier
    t = user_text.lower()
    if re.search(r"(restart|重新开始|retake|重来)", t):
        return {"intent": "restart", "extracted_value": None, "reply": "Sure — let's restart the questionnaire.", "source": "mock"}
    if re.search(r"(export|pdf|下载|download|report)", t):
        return {"intent": "export_pdf", "extracted_value": None, "reply": "I'll prepare a PDF for you.", "source": "mock"}
    m = re.search(r"\ba\s*(?:=|＝|to|为|改成|set\s+to|change\s+to)\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        return {"intent": "change_A", "extracted_value": float(m.group(1)), "reply": f"Switching to A={m.group(1)}.", "source": "mock"}
    if re.search(r"(sharpe|夏普|variance|方差|risk|回撤|drawdown|utility)", t):
        return {"intent": "ask_metric_detail", "extracted_value": None, "reply": "Let me explain that metric.", "source": "mock"}
    if re.search(r"fund[_\s]?\d+|基金", t):
        return {"intent": "ask_fund_detail", "extracted_value": None, "reply": "Let me look up that fund.", "source": "mock"}
    # Default: treat as freeform open question (caller decides whether to
    # actually call the LLM for it).
    return {"intent": "freeform", "extracted_value": None, "reply": "", "source": "mock"}


__all__ = [
    "ParseResult",
    "parse_answer",
    "generate_profile",
    "explain_portfolio",
    "classify_followup",
    "freeform_chat",
    "set_llm_enabled",
    "is_llm_active",
]


# ---------------------------------------------------------------------------
# Free-form grounded chat — used in RECOMMEND phase when the user asks open
# questions that don't match any structured intent.
# ---------------------------------------------------------------------------

def freeform_chat(
    user_text: str,
    context: dict[str, Any],
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Ask the LLM a free-form question grounded in the user's portfolio.

    ``context`` must include:
        A_value, level_code, level_name, total_score,
        weights (dict[str,float]), fund_names (dict[str,str]),
        metrics (dict with expected_return, std, sharpe, utility),
        data_source (str)

    Returns (reply_text, source) where source ∈ {"llm", "mock"}.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return ("Ask me anything about your current portfolio or risk profile.", "mock")

    # Try LLM path first
    try:
        prompt = FREEFORM_USER_PROMPT.format(
            A_value=context.get("A_value", "?"),
            level_code=context.get("level_code", "?"),
            level_name=context.get("level_name", "?"),
            total_score=context.get("total_score", "?"),
            top_holdings_block=format_top_holdings(
                context.get("weights", {}),
                context.get("fund_names", {}),
            ),
            expected_return_pct=f"{context['metrics']['expected_return'] * 100:.2f}%",
            std_pct=f"{context['metrics']['std'] * 100:.2f}%",
            sharpe=float(context["metrics"].get("sharpe", 0.0)),
            utility=float(context["metrics"].get("utility", 0.0)),
            data_source=context.get("data_source", "simulated"),
            history_block=format_history_block(history or []),
            user_text=user_text,
        )
    except Exception as exc:
        logger.warning("freeform prompt formatting failed: %s", exc)
        prompt = None

    raw = _chat(prompt, expect_json=False, temperature=0.6,
                system_prompt=FREEFORM_SYSTEM_PROMPT) if prompt else None
    if raw and len(raw.strip()) > 20:
        return (raw.strip(), "llm")

    # Mock fallback — keep it generic but grounded in what we do have.
    return (_mock_freeform(user_text, context), "mock")


def _mock_freeform(user_text: str, context: dict[str, Any]) -> str:
    A = context.get("A_value", "?")
    lvl = context.get("level_code", "?")
    weights = context.get("weights", {}) or {}
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:2]
    top_str = ", ".join(f"**{c}** {w * 100:.1f}%" for c, w in top if w >= 0.01) or "(no dominant holding)"
    exp = context.get("metrics", {}).get("expected_return", 0) * 100
    sd = context.get("metrics", {}).get("std", 0) * 100
    return (
        f"Based on your profile (**{lvl}**, **A = {A}**), your current portfolio "
        f"leans on {top_str} with about **{exp:.2f}%** expected annual return and "
        f"**{sd:.2f}%** volatility. I don't have a live LLM connection right now, "
        f"so I can only answer structured commands — try **what if A = 2**, "
        f"**explain Sharpe**, or **export pdf**. Switch on the LLM toggle in the "
        f"header for a richer conversation."
    )
