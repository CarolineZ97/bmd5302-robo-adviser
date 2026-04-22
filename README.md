# BMD5302 Robo-Adviser AI Chatbot (Part 3 / The Platform)

> **Live demo:** ➡️ _will be filled in after first deploy_
> Course: BMD5302 Financial Modeling — Group Project
> Scope: Part 3 deliverable — the user-facing platform wrapping Part 1 (Efficient Frontier) and Part 2 (Risk Aversion & Optimal Portfolio).

A Streamlit-based conversational Robo-Adviser that:

1. **AI psychological profiling** — replaces the 10-question paper questionnaire with a natural-language dialogue, scores answers via LLM, detects and clarifies self-contradictions, and translates the final risk-aversion coefficient `A` into a readable investor persona.
2. **Real-time, dynamic portfolio optimization** — pulls live fund prices (with offline fallback to Part 1's simulated data), recomputes μ and Σ, and solves the Markowitz utility `U = r - Aσ²/2` via SciPy SLSQP (matching Excel Solver output).
3. **LLM explanation layer** — every recommended portfolio is accompanied by a natural-language rationale, and the user can freely ask follow-up questions ("what if A = 2?", "why not Fund_08?") which trigger re-optimization.

## Deploy to Streamlit Community Cloud (free)

This repo is wired up for one-click deployment on <https://share.streamlit.io>:

1. Log in with your GitHub account.
2. Click **New app** → select this repository → `main` branch → main file: `app.py`.
3. Under **Advanced settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example` (filled with your real LLM key). Skip this to run in offline Mock mode.
4. Click **Deploy**. First build takes ~3 minutes.

See [`DEPLOY.md`](./DEPLOY.md) for a walkthrough.

## Quick start (local)

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Or use the provided helper:

```powershell
.\run.ps1
```

## LLM configuration (optional)

The chatbot works out-of-the-box in **Mock mode** (rule-based NLU) with no API key.
For richer natural-language understanding, use either:

- **Local:** copy `.env.example` to `.env` and fill in any OpenAI-compatible endpoint (DeepSeek / Qwen / OpenAI / Ollama).
- **Streamlit Cloud:** paste the same three keys (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`) into the app's **Secrets** panel.

## Project layout

```
bmd5302-robo-adviser/
├── app.py                  # Streamlit entry (chat UI + dashboard)
├── state_machine.py        # 7-state conversation flow
├── engine.py               # Pure-function Markowitz engine (SciPy SLSQP)
├── data_loader.py          # yfinance + CSV cache + offline fallback
├── llm_client.py           # OpenAI-compatible LLM + rule-based mock
├── prompts.py              # All LLM prompt templates
├── config.py               # Questionnaire, score maps, fund map, constants
├── visuals.py              # Plotly figure factories
├── exporter.py             # PDF investment-advice report
├── tests/test_engine.py    # Cross-check Python engine vs Excel Solver
├── data/
│   ├── fallback_prices.csv # Part 1's 60-month simulated prices
│   └── part2_baseline.json # Part 2 Excel results for regression test
├── .streamlit/
│   ├── config.toml         # Theme + server settings
│   └── secrets.toml.example
├── requirements.txt
├── packages.txt            # apt deps on Streamlit Cloud
├── runtime.txt             # Python version pin (3.12)
└── run.ps1
```

## Testing

```powershell
python -m pytest tests/ -v
```

The test suite verifies the Python engine reproduces Excel Solver's outputs for A ∈ {1,2,4,6,8} with < 0.01% error.

## Known limitations

- FSMOne has no public API; real data is pulled via Yahoo Finance proxies. Funds without a Yahoo ticker fall back to Part 1 simulated data.
- Streamlit session state is per-tab and resets on reload.
