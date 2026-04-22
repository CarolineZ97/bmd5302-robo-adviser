"""
config.py — All project-wide constants for the Robo-Adviser chatbot.

Everything that must stay in sync with the Part 2 Excel workbook lives here:
questionnaire text, option scores, question weights, score-to-risk-level table
and the A-value mapping.  Tweaking these is the only place you should need to
touch when the instructor changes the questionnaire.

Also configures the 10 fund tickers used by data_loader.py (FSMOne has no
public API so we proxy through Yahoo Finance; anything that can't be pulled
falls back to the simulated prices from Part 1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "cache.csv"
FALLBACK_FILE = DATA_DIR / "fallback_prices.csv"
BASELINE_FILE = DATA_DIR / "part2_baseline.json"

# Load .env if present (does not override existing env vars).
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _read_secret(key: str, default: str = "") -> str:
    """Read a config value, preferring Streamlit Cloud secrets when available.

    Priority:
      1. OS environment variable (works locally with .env and in CI).
      2. streamlit.secrets (works on Streamlit Community Cloud).
      3. Provided default.
    """
    val = os.getenv(key, "")
    if val:
        return val.strip()
    try:
        import streamlit as _st  # type: ignore
        # st.secrets raises if secrets.toml is missing; guard it.
        if key in _st.secrets:
            return str(_st.secrets[key]).strip()
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# LLM configuration (OpenAI-compatible)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    api_key: str = _read_secret("OPENAI_API_KEY")
    base_url: str = _read_secret("OPENAI_BASE_URL")
    model: str = _read_secret("LLM_MODEL") or "deepseek-chat"
    timeout: int = 15
    max_retries: int = 1

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url)


LLM_CONFIG = LLMConfig()


# ---------------------------------------------------------------------------
# Fund universe  (10 funds, mapped from FSMOne selection to Yahoo tickers)
# ---------------------------------------------------------------------------
# NOTE: FSMOne funds don't always have a Yahoo counterpart.  Here we pick
# 10 widely-held Singapore-domiciled mutual funds whose ISINs do resolve on
# Yahoo Finance.  If any of these fail at runtime, data_loader.py silently
# falls back to the Part 1 simulated prices.
FUND_MAP: dict[str, dict[str, str]] = {
    "Fund_01": {"yahoo": "SPY",  "name": "SPDR S&P 500 ETF (Equity/US LargeCap)"},
    "Fund_02": {"yahoo": "AGG",  "name": "iShares Core US Aggregate Bond (Fixed Income)"},
    "Fund_03": {"yahoo": "VEA",  "name": "Vanguard Developed Markets (Intl Equity)"},
    "Fund_04": {"yahoo": "VWO",  "name": "Vanguard Emerging Markets (EM Equity)"},
    "Fund_05": {"yahoo": "QQQ",  "name": "Invesco Nasdaq 100 (US Tech)"},
    "Fund_06": {"yahoo": "IWM",  "name": "iShares Russell 2000 (US SmallCap)"},
    "Fund_07": {"yahoo": "ARKK", "name": "ARK Innovation ETF (Disruptive Growth)"},
    "Fund_08": {"yahoo": "GLD",  "name": "SPDR Gold Trust (Commodity)"},
    "Fund_09": {"yahoo": "VNQ",  "name": "Vanguard Real Estate ETF (REIT)"},
    "Fund_10": {"yahoo": "EMB",  "name": "iShares EM USD Bond (EM Debt)"},
}

FUND_CODES = list(FUND_MAP.keys())

# How many months of history to use.  Matches Part 1 / 2.
HISTORY_MONTHS = 60

# Cache TTL in seconds.  1 hour is enough for a course demo.
CACHE_TTL_SECONDS = 3600


# ---------------------------------------------------------------------------
# Questionnaire (Part 2 verbatim)
# ---------------------------------------------------------------------------
# Each question: id, text, category, options list [(label, text, score)], weight.
# Weight:   Q1..Q5 -> 1,  Q6..Q10 -> 2  (Part 2 rule).

QUESTIONS: list[dict] = [
    {
        "id": 1,
        "category": "age",
        "weight": 1,
        "text": "What is your age?",
        "options": [
            ("A", "18-30", 4),
            ("B", "31-50", 3),
            ("C", "51-60", 2),
            ("D", "Above 60", 1),
        ],
    },
    {
        "id": 2,
        "category": "income",
        "weight": 1,
        "text": "What is your annual household income (SGD)?",
        "options": [
            ("A", "Below S$60,000", 1),
            ("B", "S$60,000 - 120,000", 2),
            ("C", "S$120,000 - 180,000", 3),
            ("D", "S$180,000 - 300,000", 4),
            ("E", "Above S$300,000", 5),
        ],
    },
    {
        "id": 3,
        "category": "investable_ratio",
        "weight": 1,
        "text": "What proportion of your annual household income is available for investment (excluding savings)?",
        "options": [
            ("A", "Less than 10%", 1),
            ("B", "10% to 25%", 2),
            ("C", "25% to 50%", 3),
            ("D", "More than 50%", 4),
        ],
    },
    {
        "id": 4,
        "category": "experience_breadth",
        "weight": 1,
        "text": "Which best describes your current investment experience?",
        "options": [
            ("A", "Mostly bank deposits / government bonds only", 1),
            ("B", "Mostly deposits/bonds, small portion in equities/funds", 2),
            ("C", "Diversified across deposits, bonds, fixed income, equities, funds", 3),
            ("D", "Mostly equities/funds/FX, small portion in deposits/bonds", 4),
        ],
    },
    {
        "id": 5,
        "category": "experience_years",
        "weight": 1,
        "text": "How many years of experience do you have with higher-risk products (equities, funds, FX, derivatives)?",
        "options": [
            ("A", "None", 1),
            ("B", "Less than 2 years", 2),
            ("C", "2 - 5 years", 3),
            ("D", "5 - 8 years", 4),
            ("E", "More than 8 years", 5),
        ],
    },
    {
        "id": 6,
        "category": "risk_attitude",
        "weight": 2,
        "text": "Which best describes your investment attitude?",
        "options": [
            ("A", "Highly risk-averse; no principal loss; stable returns only", 1),
            ("B", "Conservative; no principal loss; tolerate return fluctuation", 2),
            ("C", "Seek growth; willing to bear limited principal loss", 3),
            ("D", "Aim for highest return; willing to bear large principal loss", 4),
        ],
    },
    {
        "id": 7,
        "category": "prospect_theory",
        "weight": 2,
        "text": "Which lottery would you pick?",
        "options": [
            ("A", "100% chance of S$5,000", 1),
            ("B", "50% chance of S$20,000", 2),
            ("C", "25% chance of S$80,000", 3),
            ("D", "10% chance of S$200,000", 5),
        ],
    },
    {
        "id": 8,
        "category": "horizon",
        "weight": 2,
        "text": "What is your investment horizon?",
        "options": [
            ("A", "Less than 1 year", 1),
            ("B", "1 - 3 years", 2),
            ("C", "3 - 5 years", 3),
            ("D", "More than 5 years", 5),
        ],
    },
    {
        "id": 9,
        "category": "objective",
        "weight": 2,
        "text": "What is your primary investment objective?",
        "options": [
            ("A", "Capital preservation", 1),
            ("B", "Steady capital growth", 3),
            ("C", "Rapid capital growth", 5),
        ],
    },
    {
        "id": 10,
        "category": "loss_tolerance",
        "weight": 2,
        "text": "At what level of loss would you start feeling clearly anxious?",
        "options": [
            ("A", "No loss, but returns below expectation", 1),
            ("B", "A slight loss of principal", 2),
            ("C", "Up to 10% loss of principal", 3),
            ("D", "20% - 50% loss of principal", 4),
            ("E", "More than 50% loss of principal", 5),
        ],
    },
]


def get_question(q_id: int) -> dict:
    return next(q for q in QUESTIONS if q["id"] == q_id)


def score_option(q_id: int, label: str) -> int:
    q = get_question(q_id)
    for lbl, _, s in q["options"]:
        if lbl == label:
            return s
    raise ValueError(f"Unknown option {label!r} for Q{q_id}")


def weighted_total(scores_by_qid: dict[int, int]) -> int:
    """scores_by_qid: {1: 3, 2: 4, ...}.  Returns final weighted total."""
    total = 0
    for q in QUESTIONS:
        s = scores_by_qid.get(q["id"], 0)
        total += s * q["weight"]
    return total


# Score-to-risk-level mapping  (Part 2 verbatim)
RiskLevel = Literal["R1", "R2", "R3", "R4", "R5"]

SCORE_BANDS: list[tuple[RiskLevel, str, int, int]] = [
    ("R1", "Conservative Investor",        15, 27),
    ("R2", "Moderately Conservative",      28, 39),
    ("R3", "Balanced Investor",            40, 51),
    ("R4", "Growth Investor",              52, 63),
    ("R5", "Aggressive Investor",          64, 75),
]

LEVEL_TO_A: dict[RiskLevel, float] = {
    "R1": 8.0,
    "R2": 6.0,
    "R3": 4.0,
    "R4": 2.0,
    "R5": 1.0,
}


def classify_level(total_score: int) -> tuple[RiskLevel, str]:
    for code, name, lo, hi in SCORE_BANDS:
        if lo <= total_score <= hi:
            return code, name
    # Below R1 or above R5 — clamp gracefully.
    if total_score < 15:
        return "R1", "Conservative Investor"
    return "R5", "Aggressive Investor"


# ---------------------------------------------------------------------------
# Conflict detection rules  (hard-coded — no LLM needed)
# ---------------------------------------------------------------------------
# Each rule: pair of (question_id, set_of_labels).  If BOTH sides match, the
# answers are considered internally inconsistent and the chatbot will ask a
# follow-up clarification question before locking the scores in.

@dataclass(frozen=True)
class ConflictRule:
    id: str
    description: str
    left: tuple[int, frozenset[str]]   # (question_id, {"A","B"})
    right: tuple[int, frozenset[str]]


CONFLICT_RULES: list[ConflictRule] = [
    ConflictRule(
        id="risk_attitude_vs_lottery",
        description=(
            "You said you are highly risk-averse (Q6), but you also picked a "
            "long-shot lottery (Q7). Could you clarify which better reflects "
            "your real preference?"
        ),
        left=(6, frozenset({"A", "B"})),
        right=(7, frozenset({"C", "D"})),
    ),
    ConflictRule(
        id="loss_tolerance_vs_objective",
        description=(
            "You indicated you cannot tolerate any principal loss (Q10), yet "
            "you chose rapid capital growth as your objective (Q9). Rapid "
            "growth usually entails drawdowns — want to reconsider?"
        ),
        left=(10, frozenset({"A", "B"})),
        right=(9, frozenset({"C"})),
    ),
    ConflictRule(
        id="experience_vs_lottery",
        description=(
            "Your stated experience (Q4/Q5) looks entry-level, but you picked "
            "a high-variance payoff in Q7. We want to make sure you are "
            "comfortable with that level of risk."
        ),
        left=(4, frozenset({"A"})),
        right=(7, frozenset({"D"})),
    ),
    ConflictRule(
        id="horizon_vs_objective",
        description=(
            "You picked a very short investment horizon (Q8) but also rapid "
            "capital growth (Q9). Short horizons rarely permit aggressive "
            "growth — which one matters more?"
        ),
        left=(8, frozenset({"A"})),
        right=(9, frozenset({"C"})),
    ),
]


def find_conflicts(answers: dict[int, str]) -> list[ConflictRule]:
    hits: list[ConflictRule] = []
    for rule in CONFLICT_RULES:
        lq, lset = rule.left
        rq, rset = rule.right
        if answers.get(lq) in lset and answers.get(rq) in rset:
            hits.append(rule)
    return hits


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CACHE_FILE",
    "FALLBACK_FILE",
    "BASELINE_FILE",
    "LLM_CONFIG",
    "FUND_MAP",
    "FUND_CODES",
    "HISTORY_MONTHS",
    "CACHE_TTL_SECONDS",
    "QUESTIONS",
    "get_question",
    "score_option",
    "weighted_total",
    "SCORE_BANDS",
    "LEVEL_TO_A",
    "classify_level",
    "CONFLICT_RULES",
    "ConflictRule",
    "find_conflicts",
]
