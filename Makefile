PY := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)

.PHONY: dev test test-fast lint synthetic build docker

dev:            ## start API + worker + frontend
	./dev.sh

test:           ## full suite (renders a synthetic broadcast; needs ffmpeg)
	cd backend && ../$(PY) -m pytest

test-fast:      ## unit tests only, no video rendering
	cd backend && ../$(PY) -m pytest -m "not video and not live"

lint:
	cd backend && ../$(PY) -m ruff check .

synthetic:      ## render the default synthetic broadcast into inbox/
	$(PY) tools/make_synthetic_game.py --out inbox/synthetic_game.mp4

build:          ## production build of the frontend (served by the API)
	cd frontend && npm run build

docker:
	docker compose up --build
