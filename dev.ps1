# One command to run Possession Cut on Windows: API + worker + frontend.
# First run creates the virtualenv and installs dependencies.
#   .\dev.ps1            (or double-click dev.cmd)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating virtualenv (.venv)..."
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv } else { python -m venv .venv }
}

& $py -c "import possession_cut, fastapi, rapidocr_onnxruntime" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing backend dependencies..."
    & $py -m pip install --quiet --upgrade pip
    & $py -m pip install --quiet -e "backend[dev]"
}

if ((Test-Path "frontend\package.json") -and -not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..."
    Push-Location frontend
    npm install --no-fund --no-audit
    Pop-Location
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add your ANTHROPIC_API_KEY."
}

& $py -m possession_cut.dev @args
exit $LASTEXITCODE
