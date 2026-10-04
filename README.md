# Possession Cut

Local-first web app that turns a full game broadcast into a vertical (9:16) video of
every scoring possession by one team, with all dead time removed. You give it a file,
it reads the on-screen score bug, and it cuts the tape.

The app only processes video files you already have. It never downloads footage.

> Full spec: [PRD.md](PRD.md) · What was built and what to test first: [BUILD_NOTES.md](BUILD_NOTES.md)

## Setup

1. Install **ffmpeg 6+** (`winget install Gyan.FFmpeg`, `brew install ffmpeg`, or `apt install ffmpeg`).
2. Install **Python 3.12** and **Node 20+**.
3. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
4. Run it:
   - Windows: `.\dev.cmd` (or `.\dev.ps1`)
   - Mac / Linux: `./dev.sh` (or `make dev`)
5. Open <http://localhost:5173>.

The first run creates `.venv`, installs the backend and frontend dependencies, then
starts three processes: the API (`:8000`), the worker, and the frontend (`:5173`).

## Using it

1. **New job**: pick a game file (or drop one in `inbox/`), choose the sport, find the game, pick the team and a start point.
2. **Calibrate**: confirm the detected score bug and the crop preview. Saved as a reusable broadcaster template.
3. **Analyze**: watch the progress bar. A 2.5 hour broadcast takes a few minutes.
4. **Review**: toggle clips, nudge in/out points, check warnings.
5. **Export**: set the title, render, and find the MP4 in `exports/` with `cutlist.json` and `caption.txt` beside it.

## Tests

```bash
make test        # full suite, renders a synthetic broadcast (needs ffmpeg)
make test-fast   # unit tests only
```

On Windows without `make`: `cd backend; ..\.venv\Scripts\python -m pytest`.

## Docker

```bash
docker compose up --build
```

Then open <http://localhost:8000>. Game files go in `./inbox`, results land in `./exports`.
