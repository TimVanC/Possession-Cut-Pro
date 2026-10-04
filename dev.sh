#!/usr/bin/env bash
# One command to run Possession Cut: API + worker + frontend.
# First run creates the virtualenv and installs dependencies.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

if [ ! -x "$PY" ]; then
  echo "Creating virtualenv (.venv)..."
  if command -v python3.12 >/dev/null 2>&1; then python3.12 -m venv .venv
  elif command -v py >/dev/null 2>&1; then py -3.12 -m venv .venv
  else python3 -m venv .venv; fi
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
fi

if ! "$PY" -c "import possession_cut, fastapi, rapidocr_onnxruntime" >/dev/null 2>&1; then
  echo "Installing backend dependencies..."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e "backend[dev]"
fi

if [ -f frontend/package.json ] && [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install --no-fund --no-audit)
fi

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example. Add your ANTHROPIC_API_KEY."; }

exec "$PY" -m possession_cut.dev "$@"
