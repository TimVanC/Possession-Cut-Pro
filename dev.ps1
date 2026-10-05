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

# Without the dev server the engine serves the built page itself (http://127.0.0.1:8000).
# Build it when it is missing or older than the source.
if (($args -contains "--no-frontend") -and (Test-Path "frontend\package.json") -and (Get-Command npm -ErrorAction SilentlyContinue)) {
    $page = "frontend\dist\index.html"
    $stale = -not (Test-Path $page)
    if (-not $stale) {
        $built = (Get-Item $page).LastWriteTime
        $stale = [bool](Get-ChildItem "frontend\src" -Recurse -File | Where-Object { $_.LastWriteTime -gt $built } | Select-Object -First 1)
    }
    if ($stale) {
        Write-Host "Building the app page..."
        Push-Location frontend
        npm run build --silent | Out-Null
        Pop-Location
    }
}

& $py -m possession_cut.dev @args
exit $LASTEXITCODE
