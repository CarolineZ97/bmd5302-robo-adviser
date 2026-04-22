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
- "freeform"              (anything else — general investment questions, commentary, requests for interpretation)

Return STRICT JSON:
{{
  "intent": "<one of the above>",
  "extracted_value": <number or string or null>,
  "reply": "<one short polite sentence acknowledging the user's request>"
}}

User message: \"\"\"{user_text}\"\"\"

Respond with JSON ONLY.
"""


# ---------------------------------------------------------------------------
# Free-form advisory chat (used when user asks open-ended questions in the
# RECOMMEND phase — "am I conservative?", "why so much gold?", "what happens
# if the Fed cuts rates?" etc).  The model gets the current portfolio as
# grounded context so it cannot invent numbers out of thin air.
# ---------------------------------------------------------------------------

FREEFORM_SYSTEM_PROMPT = (
    "You are a seasoned yet approachable wealth advisor for the BMD5302 "
    "Robo-Adviser demo. The student has a questionnaire-derived risk "
    "aversion A and an optimized Markowitz portfolio. Your job: answer "
    "their open-ended question in 2–5 sentences, grounded ONLY in the "
    "data shown below. "
    "Rules: (1) Never invent tickers, returns, correlations, or future "
    "prices. (2) Never give a directional market forecast or single-stock "
    "buy/sell call. (3) When a number is needed, quote it from the context "
    "block verbatim. (4) If the user asks for advice outside portfolio "
    "construction (e.g. tax, legal, insurance), politely redirect. "
    "(5) Keep the tone warm, concrete, and use plain English plus "
    "light finance jargon. End with one short actionable nudge "
    "(e.g. 'try changing A to 2 and compare')."
)

FREEFORM_USER_PROMPT = """\
=== Investor context ===
A (risk aversion): {A_value}
Risk level: {level_code} — {level_name}
Questionnaire score: {total_score}/75

=== Recommended portfolio ===
{top_holdings_block}

Annualized expected return: {expected_return_pct}
Annualized volatility:      {std_pct}
Sharpe ratio:                {sharpe:.2f}
Utility U = r − A·σ²/2:      {utility:.4f}
Data source: {data_source}

=== Conversation history (most recent first) ===
{history_block}

=== User's latest question ===
\"\"\"{user_text}\"\"\"

Reply directly to the user in 2–5 sentences. Do not use JSON.
"""


def format_history_block(history: list[tuple[str, str]], max_pairs: int = 4) -> str:
    """Render the last N user/assistant exchanges for the freeform prompt.

    Each item is a (role, text) tuple where role ∈ {'user', 'assistant'}.
    Most recent first, trimmed to keep prompts small.
    """
    if not history:
        return "(no prior exchanges)"
    trimmed = history[-(max_pairs * 2):][::-1]
    lines = []
    for role, text in trimmed:
        text_short = (text or "").strip().replace("\n", " ")
        if len(text_short) > 180:
            text_short = text_short[:180] + "…"
        lines.append(f"- {role}: {text_short}")
    return "\n".join(lines)





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
