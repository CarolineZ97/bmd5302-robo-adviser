"""
state_machine.py — Conversation flow controller for the Robo-Adviser chatbot.

Seven states (per the plan):

  WELCOME          -> initial greeting, waiting for user to start
  QUESTIONNAIRE    -> asking the next pending question; parses user text
  CONFIRM_ANSWER   -> waiting for user to confirm/correct a low-confidence parse
  CONFLICT_CHECK   -> asking a follow-up to resolve an internal contradiction
  PROFILE          -> showing the investor persona, waiting for "continue"
  RECOMMEND        -> showing the recommended portfolio, accepting What-if follow-ups
  EXPORT           -> user-requested PDF export

The state machine is the ONLY layer that calls both the LLM (for NLU/NLG) and
the engine (for optimization); app.py just pumps user messages in and renders
the bot's textual + visual replies.

Design notes
------------
* `SessionState` is serializable (stored inside st.session_state).
* All state transitions are explicit in ``handle_user_input``; no hidden side
  effects — this makes the flow debuggable via pytest.
* Bot replies are returned as a list of "segments", each carrying an optional
  side-panel payload (so app.py knows when to refresh the dashboard tabs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from config import (
    LEVEL_TO_A,
    QUESTIONS,
    classify_level,
    find_conflicts,
    get_question,
    score_option,
    weighted_total,
)
from data_loader import fund_display_name, load_fund_prices
from engine import (
    compute_mu_sigma,
    efficient_frontier,
    gmvp,
    optimize_portfolio,
    backtest,
)
from llm_client import (
    classify_followup,
    explain_portfolio,
    generate_profile,
    parse_answer,
)

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    WELCOME = "welcome"
    QUESTIONNAIRE = "questionnaire"
    CONFIRM_ANSWER = "confirm_answer"
    CONFLICT_CHECK = "conflict_check"
    PROFILE = "profile"
    RECOMMEND = "recommend"
    EXPORT = "export"


@dataclass
class BotSegment:
    """One paragraph/card emitted by the bot."""

    text: str
    kind: str = "text"          # "text" | "warning" | "card" | "profile" | "portfolio"
    payload: dict[str, Any] = field(default_factory=dict)
    llm_source: str | None = None   # "llm" | "mock" | None


@dataclass
class SessionState:
    phase: Phase = Phase.WELCOME
    # questionnaire progress
    current_q: int = 1
    answers: dict[int, str] = field(default_factory=dict)    # {qid: "A"}
    scores: dict[int, int] = field(default_factory=dict)     # {qid: score}
    pending_parse: dict[str, Any] | None = None              # for CONFIRM_ANSWER
    unresolved_conflicts: list[str] = field(default_factory=list)
    # profile + recommendation
    total_score: int = 0
    level_code: str = ""
    level_name: str = ""
    A_value: float = 4.0
    profile_text: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    data_source: str = ""
    prices_cached: bool = False


# ---------------------------------------------------------------------------
# Price cache (module-level, not per-session)
# ---------------------------------------------------------------------------

_prices_df: pd.DataFrame | None = None
_mu: np.ndarray | None = None
_sigma: np.ndarray | None = None
_data_src: str = ""


def ensure_prices_loaded(force: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    global _prices_df, _mu, _sigma, _data_src
    if force or _prices_df is None:
        _prices_df, _data_src = load_fund_prices(force_refresh=force)
        _mu, _sigma = compute_mu_sigma(_prices_df, annualize=True)
    return _prices_df, _mu, _sigma, _data_src


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def greet() -> list[BotSegment]:
    return [BotSegment(
        text=(
            "Hi! I'm your **MDFinTech Robo-Adviser**. \n\n"
            "I'll ask you a few questions about your background and attitude "
            "toward risk, then build you an optimal fund portfolio based on "
            "Markowitz mean-variance theory. You can answer in natural language — "
            "there's no need to pick A/B/C/D literally.\n\n"
            "Reply **start** whenever you're ready."
        ),
        kind="text",
    )]


def handle_user_input(state: SessionState, user_text: str) -> list[BotSegment]:
    """Main dispatcher. Mutates ``state`` in place and returns bot segments."""
    user_text = (user_text or "").strip()
    if not user_text:
        return [BotSegment("Please type something so I can help.")]

    logger.info("phase=%s, text=<redacted len=%d>", state.phase.value, len(user_text))

    if state.phase == Phase.WELCOME:
        return _handle_welcome(state, user_text)
    if state.phase == Phase.QUESTIONNAIRE:
        return _handle_questionnaire(state, user_text)
    if state.phase == Phase.CONFIRM_ANSWER:
        return _handle_confirm(state, user_text)
    if state.phase == Phase.CONFLICT_CHECK:
        return _handle_conflict(state, user_text)
    if state.phase == Phase.PROFILE:
        return _handle_profile(state, user_text)
    if state.phase == Phase.RECOMMEND:
        return _handle_recommend(state, user_text)
    if state.phase == Phase.EXPORT:
        lowered = user_text.lower().strip()
        if lowered in {"restart", "重新开始", "redo"}:
            return _restart(state)
        state.phase = Phase.RECOMMEND
        return [BotSegment("Your PDF is ready via the 'Export PDF' button at the bottom.")]
    return [BotSegment("Hmm, I got confused. Let's start over — say **restart**.")]


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------

def _handle_welcome(state: SessionState, text: str) -> list[BotSegment]:
    if text.lower() in {"start", "begin", "go", "开始"}:
        state.phase = Phase.QUESTIONNAIRE
        state.current_q = 1
        return [_ask_question(1)]
    return [BotSegment("Reply **start** to begin the risk questionnaire.")]


def _ask_question(q_id: int) -> BotSegment:
    q = get_question(q_id)
    options_lines = "\n".join(f"  **{lbl}.** {text}" for lbl, text, _ in q["options"])
    weight_hint = "(weight x1)" if q["weight"] == 1 else "(weight x2)"
    text = (
        f"**Question {q_id}/10** {weight_hint}\n\n"
        f"{q['text']}\n\n"
        f"{options_lines}\n\n"
        f"_Answer with a letter, a number, or a full sentence — I'll parse it._"
    )
    return BotSegment(text=text, kind="card", payload={"q_id": q_id})


def _handle_questionnaire(state: SessionState, text: str) -> list[BotSegment]:
    q_id = state.current_q
    parsed = parse_answer(q_id, text)

    # Low-confidence or ambiguous -> ask for confirmation
    if parsed.choice is None or parsed.need_clarify or parsed.confidence < 0.5:
        state.pending_parse = {
            "q_id": q_id,
            "choice": parsed.choice,
            "score": parsed.score,
            "confidence": parsed.confidence,
            "reason": parsed.reason,
        }
        state.phase = Phase.CONFIRM_ANSWER
        if parsed.choice is None:
            return [BotSegment(
                f"I couldn't tell which option you meant. Could you be a bit more specific? "
                f"(reason: {parsed.reason})",
                kind="warning",
                llm_source=parsed.source,
            )]
        return [BotSegment(
            f"I heard you as option **{parsed.choice}** (\"{_option_text(q_id, parsed.choice)}\"). "
            f"Is that right? Reply **yes** to confirm or tell me the correct one.",
            kind="warning",
            llm_source=parsed.source,
        )]

    # Confident parse: commit answer and advance
    return _commit_answer(state, q_id, parsed.choice)


def _handle_confirm(state: SessionState, text: str) -> list[BotSegment]:
    pending = state.pending_parse or {}
    q_id = pending.get("q_id", state.current_q)
    lowered = text.lower().strip()
    if lowered in {"yes", "y", "correct", "对", "是", "没错"}:
        choice = pending.get("choice")
        if choice is None:
            # Still asking for the user's real answer
            return _handle_questionnaire(state, "yes")
        state.pending_parse = None
        return _commit_answer(state, q_id, choice)
    # User is giving a new answer - try parsing again
    state.phase = Phase.QUESTIONNAIRE
    state.pending_parse = None
    return _handle_questionnaire(state, text)


def _commit_answer(state: SessionState, q_id: int, choice: str) -> list[BotSegment]:
    state.answers[q_id] = choice
    state.scores[q_id] = score_option(q_id, choice)
    segments: list[BotSegment] = [BotSegment(
        f"Got it — Q{q_id} recorded as **{choice}** ({_option_text(q_id, choice)}).",
    )]

    # Advance
    if q_id < len(QUESTIONS):
        state.current_q = q_id + 1
        state.phase = Phase.QUESTIONNAIRE
        segments.append(_ask_question(state.current_q))
        return segments

    # All 10 questions answered — check for conflicts
    conflicts = [c for c in find_conflicts(state.answers)
                 if c.id not in state.unresolved_conflicts]
    if conflicts:
        state.phase = Phase.CONFLICT_CHECK
        rule = conflicts[0]
        state.unresolved_conflicts.append(rule.id)
        segments.append(BotSegment(
            f"Before I finalize your profile — one quick clarification:\n\n{rule.description}\n\n"
            f"Reply **'A'** if the first statement fits you better, **'B'** for the second, "
            "or tell me in your own words.",
            kind="warning",
            payload={"conflict_id": rule.id,
                     "left_q": rule.left[0], "right_q": rule.right[0]},
        ))
        return segments

    # No (more) conflicts — compute profile
    return segments + _finalize_profile(state)


def _handle_conflict(state: SessionState, text: str) -> list[BotSegment]:
    # Simple heuristic: we don't re-score here, we just note acknowledgement.
    # (A full implementation could nudge the conflicting score downward/upward.)
    segments = [BotSegment(
        "Thanks for clarifying — I'll note this in your profile for context.",
    )]
    remaining = [c for c in find_conflicts(state.answers)
                 if c.id not in state.unresolved_conflicts]
    if remaining:
        state.phase = Phase.CONFLICT_CHECK
        rule = remaining[0]
        state.unresolved_conflicts.append(rule.id)
        segments.append(BotSegment(
            f"One more: {rule.description}",
            kind="warning",
            payload={"conflict_id": rule.id},
        ))
        return segments
    # All conflicts addressed
    return segments + _finalize_profile(state)


def _finalize_profile(state: SessionState) -> list[BotSegment]:
    state.total_score = weighted_total(state.scores)
    state.level_code, state.level_name = classify_level(state.total_score)
    state.A_value = LEVEL_TO_A[state.level_code]

    subscores = {get_question(qid)["category"]: s for qid, s in state.scores.items()}
    text, src = generate_profile(
        state.total_score,
        state.level_code,
        state.level_name,
        state.A_value,
        subscores,
    )
    state.profile_text = text
    state.phase = Phase.PROFILE
    return [BotSegment(
        text=(
            f"**Risk profile complete!**\n\n"
            f"- Weighted total: **{state.total_score}/75**\n"
            f"- Risk level:   **{state.level_code} — {state.level_name}**\n"
            f"- Risk-aversion coefficient **A = {state.A_value}**\n\n"
            f"{text}\n\n"
            "Reply **continue** to see your recommended portfolio, "
            "or **restart** to redo the questionnaire."
        ),
        kind="profile",
        payload={"total_score": state.total_score, "level": state.level_code, "A": state.A_value,
                 "subscores": subscores},
        llm_source=src,
    )]


def _handle_profile(state: SessionState, text: str) -> list[BotSegment]:
    lowered = text.lower().strip()
    if lowered in {"restart", "重新开始", "redo"}:
        return _restart(state)
    if lowered in {"continue", "next", "go", "yes", "ok", "继续", "开始"}:
        return _recommend_portfolio(state)
    # Otherwise treat as a free-form follow-up
    return _handle_profile_freeform(state, text)


def _handle_profile_freeform(state: SessionState, text: str) -> list[BotSegment]:
    intent = classify_followup(text)
    if intent["intent"] == "restart":
        return _restart(state)
    return _recommend_portfolio(state)


def _recommend_portfolio(state: SessionState) -> list[BotSegment]:
    prices, mu, sigma, src = ensure_prices_loaded()
    state.data_source = src
    state.prices_cached = True
    return _run_optimization_and_reply(state, mu, sigma)


def _run_optimization_and_reply(
    state: SessionState,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> list[BotSegment]:
    result = optimize_portfolio(mu, sigma, A=state.A_value, allow_short=False)
    weights = {f"Fund_{i+1:02d}": float(w) for i, w in enumerate(result.weights)}
    metrics = {
        "expected_return": result.expected_return,
        "variance": result.variance,
        "std": result.std,
        "utility": result.utility,
        "sharpe": result.sharpe,
    }
    state.weights = weights
    state.metrics = metrics
    state.phase = Phase.RECOMMEND

    fund_names = {f"Fund_{i+1:02d}": fund_display_name(f"Fund_{i+1:02d}") for i in range(10)}
    text, llm_src = explain_portfolio(weights, metrics, state.A_value, fund_names, state.data_source)

    return [BotSegment(
        text=(
            f"**Your recommended portfolio** (A = {state.A_value}, data: {state.data_source})\n\n"
            f"- Expected annual return: **{metrics['expected_return'] * 100:.2f}%**\n"
            f"- Annual volatility: **{metrics['std'] * 100:.2f}%**\n"
            f"- Sharpe ratio: **{metrics['sharpe']:.2f}**\n"
            f"- Utility U = r - A sigma^2/2 = **{metrics['utility']:.4f}**\n\n"
            f"{text}\n\n"
            "_Ask me anything — try \"what if A = 2?\", \"why not Fund_05?\", "
            "\"export pdf\", or \"restart\"._"
        ),
        kind="portfolio",
        payload={"weights": weights, "metrics": metrics,
                 "A": state.A_value, "data_source": state.data_source},
        llm_source=llm_src,
    )]


def _handle_recommend(state: SessionState, text: str) -> list[BotSegment]:
    lowered = text.lower().strip()
    if lowered in {"restart", "重新开始", "redo"}:
        return _restart(state)

    intent = classify_followup(text)
    kind = intent["intent"]
    if kind == "restart":
        return _restart(state)
    if kind == "export_pdf":
        state.phase = Phase.EXPORT
        return [BotSegment("Click the **Export PDF** button at the bottom of the page.",
                           kind="warning")]
    if kind == "change_A":
        new_A = intent.get("extracted_value")
        if new_A and 0.5 <= float(new_A) <= 20:
            state.A_value = float(new_A)
            prices, mu, sigma, _src = ensure_prices_loaded()
            return [BotSegment(f"Re-optimizing with A = {new_A}...")] + \
                   _run_optimization_and_reply(state, mu, sigma)
        return [BotSegment(
            "I need an A value between 0.5 and 20. Try \"A = 2\" or \"set A to 6\".",
            kind="warning",
        )]
    if kind == "ask_metric_detail":
        return [BotSegment(_metric_explainer(text))]
    if kind == "ask_fund_detail":
        return [BotSegment(_fund_explainer(text, state.weights))]

    # Fallback: echo with a friendly nudge
    return [BotSegment(
        "I'm not sure what you meant. Try **what if A = 2**, **explain Sharpe**, "
        "**export pdf**, or **restart**.",
        kind="warning",
    )]


def _restart(state: SessionState) -> list[BotSegment]:
    state.phase = Phase.QUESTIONNAIRE
    state.current_q = 1
    state.answers.clear()
    state.scores.clear()
    state.pending_parse = None
    state.unresolved_conflicts = []
    return [BotSegment("Restarting the questionnaire. Here we go!"), _ask_question(1)]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _option_text(q_id: int, label: str) -> str:
    q = get_question(q_id)
    for lbl, text, _ in q["options"]:
        if lbl == label:
            return text
    return ""


def _metric_explainer(user_text: str) -> str:
    t = user_text.lower()
    if "sharpe" in t or "夏普" in t:
        return ("**Sharpe ratio** = (expected return - risk-free rate) / std dev. "
                "Higher is better; > 1 is considered good for a long-only portfolio.")
    if "variance" in t or "方差" in t or "volatility" in t or "std" in t:
        return ("**Variance / Std dev** measures how much monthly returns bounce around their "
                "average. The Markowitz utility penalizes it quadratically — that's why we "
                "look for diversification.")
    if "utility" in t:
        return ("**Utility** U = r - A*sigma^2/2. It's the single number the optimizer "
                "maximizes: it rewards return and penalizes risk in proportion to your "
                "risk-aversion coefficient A.")
    return ("I can explain Sharpe ratio, variance/volatility, or the utility formula. "
            "Ask me about any of those.")


def _fund_explainer(user_text: str, weights: dict[str, float]) -> str:
    import re as _re
    m = _re.search(r"fund[_\s]?(\d{1,2})", user_text.lower())
    if not m:
        return "Tell me which fund you're curious about — e.g. 'tell me about Fund_07'."
    idx = int(m.group(1))
    code = f"Fund_{idx:02d}"
    w = weights.get(code, 0.0)
    name = fund_display_name(code)
    if w < 0.01:
        return (f"**{code}** ({name}) currently has ~0% weight in your portfolio. "
                "The optimizer found more attractive risk-return tradeoffs elsewhere "
                "given your A value.")
    return (f"**{code}** — {name} — is allocated **{w*100:.1f}%** of your portfolio. "
            "It was selected because it pushes your risk-adjusted utility U higher than "
            "alternatives with similar volatility.")


__all__ = [
    "Phase",
    "BotSegment",
    "SessionState",
    "greet",
    "handle_user_input",
    "ensure_prices_loaded",
]
