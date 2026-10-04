# Possession Cut

Turns a full game broadcast into a vertical (9:16) video of every scoring possession by
one team, with all dead time removed. You point it at a file, it reads the on-screen
score bug, and it cuts the tape: each clip starts when the possession starts and ends
right after the make.

It runs on your own machine. It only processes video files you already have, and it
never downloads footage.

> Spec: [PRD.md](PRD.md) · What was built, decisions, known gaps, what to test first: [BUILD_NOTES.md](BUILD_NOTES.md)

## Setup

1. Install **ffmpeg 6 or newer**: `winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (Mac), `apt install ffmpeg` (Linux).
2. Install **Python 3.12** and **Node 20 or newer**.
3. Copy `.env.example` to `.env`.
4. Optional, for Claude vision calibration and written captions: set `ANTHROPIC_API_KEY` in `.env`. If the API says the key is not scoped to a workspace, also set `ANTHROPIC_WORKSPACE_ID`. Without Claude everything still works: the score bug is found by the on-device detector and captions come from a template.
5. Start it:
   - Windows: `.\dev.cmd` (or `.\dev.ps1`)
   - Mac / Linux: `./dev.sh` (or `make dev`)
6. Open <http://localhost:5173>.

The first run creates `.venv`, installs the backend and frontend dependencies, and starts
three processes: the API (`:8000`), the worker, and the frontend (`:5173`). Ctrl+C stops
all three.

## Using it

1. **New job.** Browse to a game file, or drop one in `inbox/` and open the draft that appears. MP4, MKV or TS, 480p or better. Files are read in place.
2. **Game setup.** Pick the sport, find the game by date, choose the team to follow, and choose where the cut starts: the start of the game, a game time (Q3 2:26), or automatically at the team's largest deficit.
3. **Calibrate.** The app finds the score bug and shows what it reads in each field. Drag the boxes if anything is off, then confirm. The layout is saved as a broadcaster template and reused next time.
4. **Analyze.** A 2.5 hour broadcast takes roughly 5 to 10 minutes.
5. **Review.** Every clip with its score, scorer and confidence. Toggle clips off, nudge in and out points, watch the whole cut in sequence.

   | Key | Action |
   | --- | --- |
   | `J` / `K` | previous / next clip |
   | `Space` | play / pause |
   | `X` | toggle the clip |
   | `[` / `]` | in point 0.5 s earlier / later |
   | `{` / `}` | out point 0.5 s earlier / later |

6. **Export.** Set the title and render. The MP4 lands in `exports/` with `…cutlist.json` and `…caption.txt` beside it.

## Where things live

| Path | What |
| --- | --- |
| `inbox/` | drop game files here; each becomes a draft job |
| `exports/` | finished videos, cut lists, captions |
| `data/jobs/{id}/` | per-job artifacts: `probe.json`, `calibration.json`, `timeline.parquet`, `pbp.json`, `events.json`, `cutlist.json`, thumbnails, logs |
| `data/templates/` | saved score bug layouts |
| `data/cache/` | play-by-play, cached per game |

Change the folders, the Claude model, the per-job Claude budget and the sample rate in `.env`.
Game files are large: if this repo sits in a synced folder (OneDrive, iCloud, Dropbox), point
`DATA_DIR`, `INBOX_DIR` and `EXPORTS_DIR` somewhere outside it.

## Sports

NBA is fully tested end to end. NFL, NHL and MLB adapters are implemented and unit tested
against recorded play-by-play, and still need tuning on real broadcasts.

| Sport | Play-by-play | Clip starts at | Clip ends |
| --- | --- | --- | --- |
| NBA | stats.nba.com / cdn.nba.com | the possession start: shot clock reset, clock starting after a stoppage, or the opponent scoring | score appears + 1.5 s |
| NFL | nflverse (`nflreadpy`) | the snap of the scoring play | score appears + 2 s, PAT as a tail |
| NHL | api-web.nhle.com | the last faceoff, or 15 s before the goal | score appears + 2 s |
| MLB | statsapi.mlb.com | 3 s before the final pitch | score appears + 2 s (+5 s for a home run trot) |

## Tests and tools

```bash
make test        # full suite; renders a synthetic broadcast, needs ffmpeg (about 5 minutes the first time)
make test-fast   # logic only, no video (seconds)
```

Without `make` (Windows): `cd backend` then `..\.venv\Scripts\python -m pytest`.

| Tool | What it does |
| --- | --- |
| `python tools/make_synthetic_game.py --out inbox/synthetic_game.mp4` | renders a scripted game with an ESPN-style bug, commercials, replays and free throws, plus its ground-truth cut list |
| `python tools/verify_export.py exports/<file>.mp4` | checks an export against the output rules (ffprobe, canvas, crop). Add `--truth <game>.truth.json` for a frame-by-frame check on a synthetic game |
| `python tools/benchmark_analysis.py --minutes 150 --height 1080` | times the analysis stage on a long file |

## Docker

```bash
docker compose up --build
```

Then open <http://localhost:8000>. `./inbox`, `./data` and `./exports` are mounted from the
host. To browse another folder of recordings, set `GAMES_DIR=/path/to/recordings` before
running; it appears in the file picker as `/games`.

## Hosted frontend (optional)

The frontend can be deployed on its own as a static site (`frontend/vercel.json` is set up
for Vercel with `frontend` as the root directory). It still talks to the engine running on
your computer, so:

1. Start the engine locally as above.
2. Add the site's address to `.env`: `CORS_ORIGINS=https://your-app.vercel.app`, and restart.
3. Open the site in Chrome. It looks for the engine at `http://127.0.0.1:8000`.

The backend is not meant to be hosted: it reads multi-gigabyte local files, runs ffmpeg and
a long-lived worker, and has no login.
