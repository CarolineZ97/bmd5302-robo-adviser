# BMD5302 Robo-Adviser Chatbot one-click launcher (Windows PowerShell)
# Usage: from the repo root, run:  .\run.ps1

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

# 1. Create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "[run.ps1] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

# 2. Activate venv
& .\.venv\Scripts\Activate.ps1

# 3. Install deps (quiet if already installed)
Write-Host "[run.ps1] Installing dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

# 4. Launch Streamlit
Write-Host "[run.ps1] Launching Streamlit app at http://localhost:8501 ..." -ForegroundColor Green
streamlit run app.py
