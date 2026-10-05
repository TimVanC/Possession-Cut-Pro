# Possession Cut

Turns a full game broadcast into a vertical (9:16) video of every scoring possession by
one team, with all dead time removed. You upload the game, it reads the on-screen score
bug, and it cuts the tape: each clip starts when the possession starts and ends right
after the make.

The video work is done by an engine on your own computer. It only processes video files
you give it, and it never downloads footage.

> Spec: [PRD.md](PRD.md) · What was built, decisions, known gaps, what to test first: [BUILD_NOTES.md](BUILD_NOTES.md)

## Setup

1. Install **ffmpeg 6 or newer**: `winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (Mac), `apt install ffmpeg` (Linux).
2. Install **Python 3.12** and **Node 20 or newer**.
3. Copy `.env.example` to `.env`.
4. Optional, for Claude vision calibration, re-reads of unclear frames and written captions: set `ANTHROPIC_API_KEY` in `.env`. If the API says the key is not scoped to a workspace, also set `ANTHROPIC_WORKSPACE_ID` (Claude Console, Settings, Workspaces). Without Claude everything still works: the score bug is found by the on-device detector and captions come from a template.
5. Start the engine:
   - Windows: double-click `start.cmd`
   - Mac / Linux: `./dev.sh` (or `make dev`)
6. Open <http://127.0.0.1:8000> (or <http://localhost:5173> after `dev.cmd` / `./dev.sh`, when working on the code).

The first run creates `.venv` and installs what it needs, which takes a few minutes.

**Start it automatically (Windows).** Double-click `install-autostart.cmd` once. The engine
then starts quietly in the background every time you log in, and the page just works.
`remove-autostart.cmd` undoes it. The engine's log is `data/engine.log`.

| Launcher | What it starts |
| --- | --- |
| `start.cmd` | the engine (API and worker). Leave the window open. |
| `install-autostart.cmd` | the engine, hidden, now and at every login |
| `dev.cmd` / `./dev.sh` | the engine plus the frontend dev server on `:5173`, for working on the code |

## Using it

1. **Upload.** Drop the game file on the page or click to choose it. MP4, MKV or TS, 480p or better. A 5 GB game takes about a minute. If the upload is interrupted it resumes where it stopped.
   If the file is already on the computer running the engine you can pick it from disk instead and skip the copy, or drop it in `inbox/`.
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
| uploads folder | game files received through the page. `data/uploads`, or your computer's local app-data folder when this project sits inside OneDrive, Dropbox or iCloud. Set `UPLOADS_DIR` in `.env` to choose. An uploaded file is deleted when you delete its job. |
| `inbox/` | drop game files here; each becomes a draft job |
| `exports/` | finished videos, cut lists, captions |
| `data/jobs/{id}/` | per-job artifacts: `probe.json`, `calibration.json`, `timeline.parquet`, `pbp.json`, `events.json`, `cutlist.json`, thumbnails, logs |
| `data/templates/` | saved score bug layouts |
| `data/cache/` | play-by-play, cached per game |

Change the folders, the Claude model, the per-job Claude budget and the sample rate in `.env`.
A file you pick from disk or drop in `inbox/` is read where it is and never deleted.

## Sports

NBA is tested end to end, including on a real Finals broadcast. NFL, NHL and MLB adapters
are implemented and unit tested against recorded play-by-play, and still need tuning on
real broadcasts.

| Sport | Play-by-play | Clip starts at | Clip ends |
| --- | --- | --- | --- |
| NBA | stats.nba.com / cdn.nba.com | the possession start: shot clock reset, clock starting after a stoppage, the opponent scoring, or the bug returning from a break | about 2.5 s after the ball drops, once the new score has shown |
| NFL | nflverse (`nflreadpy`) | the snap of the scoring play | score appears + 2 s, PAT as a tail |
| NHL | api-web.nhle.com | the last faceoff, or 15 s before the goal | score appears + 2 s |
| MLB | statsapi.mlb.com | 3 s before the final pitch | score appears + 2 s (+5 s for a home run trot) |

Crowd shots and player close-ups at the start or end of a clip are trimmed off ("Game
camera only" in game setup, on by default).

## Tests and tools

```bash
make test        # full suite; renders a synthetic broadcast, needs ffmpeg (about 5 minutes the first time)
make test-fast   # logic only, no video (seconds)
```

Without `make` (Windows): `cd backend` then `..\.venv\Scripts\python -m pytest`.

| Tool | What it does |
| --- | --- |
| `python tools/make_synthetic_game.py --out inbox/synthetic_game.mp4` | renders a scripted game with an ESPN-style bug, commercials, replays and free throws, plus its ground-truth cut list |
| `python tools/verify_export.py exports/<file>.mp4` | checks an export against the output rules (ffprobe, canvas, crop). Add `--truth <game>.truth.json` for a frame-by-frame check on a synthetic game, or `--calibration data/jobs/<id>/calibration.json` for real footage |
| `python tools/benchmark_analysis.py --minutes 150 --height 1080` | times the analysis stage on a long file |

## Hosted on a server (Railway)

The whole app (page, API, worker, ffmpeg) runs as one container from the `Dockerfile`, so
it works from any device with nothing installed.

Service settings (Railway dashboard, Settings): build from `Dockerfile`; pre-deploy
command `python -m possession_cut.selfcheck`; health check path `/api/health`; restart on
failure. Railway's `railway.json` config file is deprecated, so these live on the service.

Set these variables on the service:

| Variable | Value | Why |
| --- | --- | --- |
| `APP_PASSWORD` | a password you choose | The sign-in for the app. Without it a hosted copy serves nothing. |
| `ANTHROPIC_API_KEY` | your key (optional) | Claude calibration, re-reads and captions. |
| `ANTHROPIC_WORKSPACE_ID` | your workspace ID | Only if the key is not tied to a workspace. |
| `HOSTED` | `1` | Sign-in required, uploads only, no view of the server's disk. |
| `DATA_DIR` | `/data` | Job records and saved layouts, on the persistent volume. |
| `EXPORTS_DIR` | `/data/exports` | Finished videos, on the volume. |
| `UPLOADS_DIR` | `/scratch/uploads` | Uploaded games, on the temporary disk. |
| `SCRATCH_DIR` | `/scratch/work` | Big temporary files. |
| `INBOX_DIR` | `/scratch/inbox` | Unused on a server; kept off the volume. |
| `UPLOAD_RETENTION_DAYS` | `7` | An uploaded game is deleted this long after its job last changed. |
| `EXPORT_RETENTION_DAYS` | `30` | A finished video is deleted after this long. |

Attach a volume at `/data` and give the service a public domain.

What to know:

- **Uploaded games sit on the temporary disk**, because a small plan's volume (5 GB) is
  smaller than one game. A redeploy or restart wipes that disk: the job stays, but the
  game has to be uploaded again to keep working on it. Finish and download a cut before
  deploying new code. A plan with a bigger volume can point `UPLOADS_DIR` at `/data/uploads`.
- **Old exports are removed to make room** when the volume is nearly full. Download the
  ones you want to keep.
- **Deploying new code.** In Railway press Ctrl+K and choose **Deploy Latest Commit**.
  For pushes to deploy by themselves, the Railway GitHub App needs access to the
  repository (GitHub, Settings, Applications, Railway, Configure). Only changes under
  `backend/`, `frontend/`, `tools/` or to the `Dockerfile` redeploy; a docs-only push does not.
- **Which build is live** is shown as `build` at `/api/health`.

## The Vercel address

`frontend/vercel.json` makes the Vercel site forward every address to the hosted app on
Railway, so either link opens the same thing. The redirect is temporary (307), so
changing it takes effect at once.

To run a static page that talks to an engine on your own computer instead, replace the
`redirects` block with `"rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]`,
add the site's address to `.env` as `CORS_ORIGINS=https://your-app.vercel.app`, and start
the engine. On the computer the engine runs on, <http://127.0.0.1:8000> needs none of that.

## Docker

```bash
docker compose up --build
```

Then open <http://localhost:8000>. `./inbox`, `./data` and `./exports` are mounted from the
host. To browse another folder of recordings, set `GAMES_DIR=/path/to/recordings` before
running; it appears in the file picker as `/games`. Not yet tested on a machine with Docker.
