"""
prompts.py — All LLM prompt templates, centralized.

Three high-level tasks:

1. parse_answer      — convert a user's free-form answer into a structured option pick.
2. generate_profile  — turn the numeric score + A value into a readable investor persona.
3. explain_portfolio — articulate WHY the optimizer chose these weights, in plain English.

Conventions:

* All templates expect a ``.format(**kwargs)`` call.
* LLM must return STRICT JSON where structured output is required — we parse
  defensively on the Python side but we tell the model exactly what shape to
  emit.
* We keep prompts terse (< 400 tokens) to minimize API latency and cost.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are the NLU/NLG layer of a Robo-Adviser built for the BMD5302 "
    "Financial Modeling project. You DO NOT perform any numerical "
    "optimization — the Python engine does that. Your job is to (a) map "
    "free-form user answers onto the fixed multiple-choice options of a "
    "risk questionnaire, and (b) explain the engine's numerical output in "
    "natural language. Be precise, warm, and avoid financial jargon unless "
    "the user uses it first."
)

# ---------------------------------------------------------------------------
# 1. Parse free-form questionnaire answer
# ---------------------------------------------------------------------------

PARSE_ANSWER_PROMPT = """\
You are matching a user's answer to one of the predefined options below.

Question (Q{q_id}): {question_text}

Options:
{options_block}

User's answer (free-form):
\"\"\"{user_text}\"\"\"

Return STRICT JSON with this shape:
{{
  "choice": "A" | "B" | "C" | "D" | "E" | null,
  "confidence": <float 0.0-1.0>,
  "need_clarify": <bool>,
  "reason": "<one short sentence, why you picked that option>"
}}

Rules:
- If the user's answer maps cleanly to one option, return that letter with confidence >= 0.7.
- If the answer is ambiguous, irrelevant, or unparseable, set choice=null, need_clarify=true.
- Respond with JSON ONLY, no markdown fence, no commentary.
"""

# ---------------------------------------------------------------------------
# 2. Investor-persona report
# ---------------------------------------------------------------------------

PROFILE_REPORT_PROMPT = """\
A user has completed a risk-tolerance questionnaire. Their results:

- Weighted total score: {total_score} / 75
- Risk level: {level_code} ({level_name})
- Risk-aversion coefficient A: {A_value}
- Per-category sub-scores: {subscores_json}

Write a 120-150 word investor persona in English. Structure:
1. One sentence naming the persona (e.g. "You are a balanced, growth-oriented investor").
2. 2-3 sentences on their strengths: what kind of market conditions suit them.
3. 1-2 sentences on pitfalls to watch out for.
4. Finish with: "Your recommended risk-aversion coefficient A = {A_value}."

Be warm and direct. Do NOT invent new numbers. Do NOT suggest specific funds or
allocations — the portfolio engine handles that separately.
"""

# ---------------------------------------------------------------------------
# 3. Portfolio explanation
# ---------------------------------------------------------------------------

EXPLAIN_PORTFOLIO_PROMPT = """\
The Markowitz engine has recommended the following portfolio for a user with
risk-aversion coefficient A = {A_value}:

Top holdings (weight >= 1%):
{top_holdings_block}

Headline metrics (annualized):
- Expected return: {expected_return_pct}
- Volatility (std): {std_pct}
- Sharpe ratio:    {sharpe:.2f}
- Utility U = r - A * sigma^2 / 2: {utility:.4f}

Data source label: {data_source}  (e.g. yfinance / fallback)

Write 4-6 short bullet points (Markdown) that explain, in plain English:
1. Why the engine chose this mix given their A value.
2. What each of the top 2-3 holdings contributes (diversification, growth, defense).
3. What the user should expect in a bad month vs a good month.
4. One honest caveat (past performance, data source limitations, etc.).

Do NOT invent numbers that are not in the inputs above. Keep total length under
180 words. Use a friendly, adviser-like tone.
"""

# ---------------------------------------------------------------------------
# 4. Free-form follow-up (What-If questions)
# ---------------------------------------------------------------------------

FOLLOWUP_INTENT_PROMPT = """\
The user has a recommended portfolio and is asking a follow-up question. Classify
their intent into ONE of:

- "change_A"              (they want to change their risk aversion — extract new A value if given)
- "change_horizon"        (they want to change investment horizon)
- "ask_fund_detail"       (they want info about a specific fund)
- "ask_metric_detail"     (they want explanation of a metric like Sharpe / variance)
- "restart"               (they want to retake the questionnaire)
- "export_pdf"            (they want to export / download the report)
- "unknown"

Return STRICT JSON:
{{
  "intent": "<one of the above>",
  "extracted_value": <number or string or null>,
  "reply": "<one short polite sentence acknowledging the user's request>"
}}

User message: \"\"\"{user_text}\"\"\"

Respond with JSON ONLY.
"""


def format_options_block(options: list[tuple[str, str, int]]) -> str:
    """Format the 4/5 options for embedding into PARSE_ANSWER_PROMPT."""
    return "\n".join(f"  {lbl}. {text}  (score={score})" for lbl, text, score in options)


def format_top_holdings(weights: dict[str, float], fund_names: dict[str, str]) -> str:
    """Format top holdings for EXPLAIN_PORTFOLIO_PROMPT."""
    lines = []
    for code, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if w >= 0.01:
            name = fund_names.get(code, code)
            lines.append(f"  - {code} ({name}): {w * 100:.2f}%")
    return "\n".join(lines) if lines else "  (no holdings above 1%)"


__all__ = [
    "SYSTEM_PROMPT",
    "PARSE_ANSWER_PROMPT",
    "PROFILE_REPORT_PROMPT",
    "EXPLAIN_PORTFOLIO_PROMPT",
    "FOLLOWUP_INTENT_PROMPT",
    "format_options_block",
    "format_top_holdings",
]
