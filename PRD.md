# Possession Cut: Auto Game-Run Clipper PRD

Oct 4, 2026 · @Tim Van Cauwenberge

## Overview

Possession Cut is a local-first web app that turns a full game broadcast into a vertical short-form video showing every scoring possession by one team from a chosen start point, with all dead time removed.

**Reference format:** a TikTok of the Knicks' 29-point comeback vs the Spurs in the NBA Finals. It starts at the possession that kicked off the run and shows every Knicks score in sequence. Each clip begins when the possession starts and ends right after the make. No timeouts, free-throw walk-ups, replays, crowd shots, or commercials. Result: about 5 minutes of constant action from a 2.5-hour broadcast.

**Goal for this build:** a complete v2 web app (not a CLI prototype) that Tim can run on his own machine: upload a game file, fill a short form, review the auto-generated cut list, and export a finished MP4. It must work for NBA first and be structured so NFL, NHL, and MLB plug in as sport adapters.

**Primary user:** Tim (single user, technical). No auth, no multi-tenancy.

**Core principle:** the on-screen scoreboard (score bug) is the source of truth for timing. League play-by-play data labels and cross-checks the clips; it does not drive the cuts.

## User flow

1. **New job.** Tim selects a local video file (MP4/MKV/TS, up to \~15 GB, 720p or 1080p). Two input paths: a file picker that registers the file by path (no browser upload of multi-GB files), and a watched `inbox/` folder where any dropped file appears as a draft job.
2. **Game setup form.**
   - Sport (NBA default; NFL, NHL, MLB).
   - Game lookup: date + teams. The app queries the sport adapter and shows matching games to pick from, which returns the league game ID.
   - Team to follow (dropdown of the two teams).
   - Start point: game time (period + clock, e.g. Q3 2:26), or "start of game", or "auto: start of biggest run" (computed from play-by-play: the moment the followed team trailed by its maximum deficit).
   - End point: game time or "end of game" (default).
   - Optional toggles: include free throws (default on, trimmed tight), include and-ones' free throw (default on), include opponent scores (default off).
3. **Calibrate.** The app samples frames, finds the score bug automatically, and shows Tim the detected box drawn on a frame plus a preview of the crop. Tim confirms or drags the box to adjust. Calibration is saved per broadcaster template (e.g. "ESPN NBA 2026") and reused automatically next time.
4. **Analyze.** Background job with a live progress bar: frame sampling, OCR, timeline building, play-by-play fetch, matching, clip boundary detection.
5. **Review.** A clip list with thumbnails, the score before and after, scorer and play description, confidence, and an inline player. Tim can toggle clips off, nudge in/out points, and see warnings (unmatched play-by-play events, low-confidence OCR).
6. **Export.** Renders the final 9:16 MP4 with the crop and overlays, shows it in an in-app player, and saves it to `exports/`. Also exports a sidecar JSON of the cut list and a suggested caption.

## Core pipeline

The pipeline reads the score bug once or twice per second, builds a clean game timeline from it, and derives clip boundaries from score changes and possession resets. Each stage writes its output to the job folder so stages can be rerun independently and debugged.

1. **Probe.** `ffprobe` for duration, fps, resolution. Reject files under 480p.
2. **Locate the score bug (calibration).**
   - Sample \~12 full frames spread across the file. Send them to Claude vision (`claude-sonnet-5-5` via the Anthropic API) asking for the score bug bounding box and the sub-boxes for: away team label, away score, home team label, home score, period, game clock, shot clock (or sport equivalents). Require JSON output.
   - Validate with pixel statistics: the bug region is static across frames (low temporal variance on its frame and borders). Take the median box across frames.
   - Save as a reusable **broadcaster template** (normalized coordinates + sub-ROIs + a reference crop image). On new jobs, first try matching existing templates by image similarity before calling Claude.
3. **Sample and OCR.** Decode at 2 fps (configurable) with ffmpeg, cropping only the score bug ROI so decode and OCR stay fast. For each sub-ROI: upscale 3x, binarize, run digit-restricted OCR. Use RapidOCR/PaddleOCR (ONNX, CPU) as the primary engine. Frames below a confidence threshold get a batched Claude vision fallback, capped at a configurable budget per job (default $2). Target: a 2.5-hour broadcast analyzed in under 15 minutes on a laptop CPU.
4. **Build the timeline.** One row per sample: `t_video, bug_visible, period, clock, shot_clock, score_home, score_away, confidence`. Clean it:
   - Scores never decrease within a game; game clock only decreases within a period. Reject reads that violate these and fill from neighbors.
   - A new score value counts only after it holds for 2+ consecutive samples. Broadcasts often animate the bug on a score (in the reference, the Knicks wordmark briefly covers the score), so brief unreadable stretches right after a score are expected.
   - Mark each sample **live** or **not live**. Not live = bug hidden (commercials, many replays, crowd shots with graphics) or clock/score values that jump backward (a replay showing an earlier game state).
   - Mark **clock running** where the game clock decreases across consecutive samples.
5. **Detect score events.** Each stable change in the followed team's score = one event, with its point value (1, 2, 3, or sport equivalent) and the video timestamp the new score first appeared.
6. **Find clip boundaries** (NBA rules here; other sports in the sport section).
   - **End:** score-appeared time + post-roll (default 1.5 s). The bug lags the actual make by 0.5 to 2 s, so the make is already on screen.
   - **Start:** walk back from the make to the most recent possession start, which is the latest of: shot clock jumping up to 24 (or 14), game clock starting to run after a stoppage, or an opponent score change. Subtract a 1 s pre-roll.
   - **Free throws** (clock stopped, +1 changes): a short window of 3 s before to 1 s after each made FT. Consecutive FTs merge into one clip. An and-one FT attaches to its basket clip as a short trailing segment.
   - **Guards:** clip length 3 to 30 s (longer possessions get trimmed to the last 30 s before the make). Any not-live stretch inside a clip is cut out. Overlapping clips merge.
7. **Cross-check with play-by-play.** Fetch scoring events from the sport adapter (period, clock, team, points, scorer, description, running score). Match each detected event by period + clock within 3 s + team + points. Matched clips get the scorer and description as labels. Report unmatched events in both directions in the review UI. If play-by-play is unavailable (very old games, API down), the pipeline still completes from the bug alone.
8. **Auto start point (optional).** From play-by-play running score, find the followed team's largest deficit and start at the next possession after it. Show the computed start in the form before analysis runs.

## Output format

The export is a 1080x1920 (9:16) MP4 with a near-square crop of the broadcast centered on a black canvas, sized so the score bug spans the bottom of the crop almost edge to edge. This matches the reference video and is required for every sport.

**Crop rule** (measured from the reference screenshots):

- Crop height = full source height. The score bug stays at the bottom of the crop, fully visible.
- Crop width = score bug width / 0.78, so the bug fills about 78% of the crop width with even margins. In the reference this yields roughly a 1.18:1 crop from a 16:9 source.
- Horizontal center = score bug center (not frame center; some networks place the bug off-center).
- Clamp to the frame edges. Crop width and position are computed once per broadcaster template and stored with it; Tim can override in calibration.
- Scale the crop to 1080 px wide and center it vertically on the 1080x1920 black canvas.

**Overlays** (all optional, off by default except the title):

- Title text in the top black bar, e.g. "Knicks 29-point comeback vs Spurs". Editable in the export dialog. Clean bold sans font, white.
- Optional small caption in the bottom black bar (e.g. game and date).
- Nothing is drawn over the video itself.

**Cuts and audio:** hard video cuts between clips. Keep the broadcast audio, with a 60 to 100 ms audio crossfade at each cut to avoid pops.

**Encoding:** H.264 High, CRF 18, source frame rate (cap at 60), AAC 192 kbps, `+faststart`. Export via a single ffmpeg filter graph from a generated concat list (no re-encoding passes per clip unless needed for accuracy).

**Also exported:** `cutlist.json` (every clip with source in/out, game time, score, scorer, confidence) and `caption.txt` (a suggested post caption and hashtags, generated by Claude from the play-by-play).

## Sport adapters

Each sport is one adapter module implementing the same interface, so the pipeline core stays sport-agnostic. NBA ships fully tested in this build; NFL, NHL, and MLB adapters ship implemented and unit-tested against saved play-by-play fixtures, with end-to-end tuning after Tim tests real files.

**Adapter interface:** `find_games(date, team?)`, `fetch_pbp(game_id)` returning normalized scoring events, `bug_fields` (which sub-ROIs to locate), `possession_start(timeline, event)`, `default_rolls` (pre/post-roll seconds), `max_clip_seconds`.

| Sport | Play-by-play source (no key needed) | Score bug fields | Clip start rule | Clip end rule |
| --- | --- | --- | --- | --- |
| NBA | `nba_api` (stats.nba.com play-by-play v3; cdn.nba.com liveData for recent games). Needs browser-like headers and retry with backoff. | scores, period, game clock, shot clock | shot clock reset to 24/14, clock start after stoppage, or opponent score; 1 s pre-roll | score appears + 1.5 s |
| NFL | nflverse via `nflreadpy` (play-by-play 1999 to now) | scores, quarter, game clock, down and distance, play clock if shown | the scoring play's snap: clock starting to run, or the bug clock reaching the play's start time from play-by-play; 2 s pre-roll | score appears + 2 s; PAT/2-pt attempt as a short optional tail |
| NHL | NHL public API (`api-web.nhle.com/v1/gamecenter/{id}/play-by-play`) | scores, period, game clock | later of the last faceoff (clock starts) or 15 s before the goal | score appears + 2 s |
| MLB | MLB Stats API (`statsapi.mlb.com`, `MLB-StatsAPI` package) | scores, inning and half, outs, count, runners | 3 s before the final pitch of the scoring plate appearance (last count change before the score change) | score appears + 2 s; home runs get +5 s for the trot (toggle) |

**NFL extra mode (optional, if time permits):** "drive mode" includes the big plays of each scoring drive (gains of 15+ yards and third/fourth-down conversions, from play-by-play) before the scoring play, for comebacks like 28-3 where the drives are the story.

All adapters cache play-by-play responses to disk per game ID.

## Architecture

The app runs entirely on Tim's machine: a React frontend, a FastAPI backend, and a separate worker process that does the heavy video work. Game files are 3 to 15 GB, so nothing is uploaded to a server and nothing is deployed to Vercel in this version.

&#91;embedded content: system architecture · 7 parts\]

The API queues jobs and serves media; the worker does all decoding, OCR, and rendering, and is the only part that calls Claude and the league APIs.

**Stack**

| Layer | Choice |
| --- | --- |
| Frontend | React + TypeScript + Vite, Tailwind, TanStack Query |
| API | Python 3.12, FastAPI, uvicorn, Pydantic v2 |
| Worker | separate Python process polling a jobs table; one job at a time; progress written to the DB |
| Storage | SQLite (SQLModel) for jobs, templates, clips, exports; per-job folder for artifacts |
| Video | ffmpeg/ffprobe 6+ via subprocess; OpenCV (headless) and NumPy for frame stats |
| OCR | RapidOCR (ONNX runtime, CPU); Claude vision fallback |
| AI | `anthropic` Python SDK: calibration, low-confidence OCR fallback, caption text |
| League data | `nba_api`, `nflreadpy`, `MLB-StatsAPI`, `httpx` for NHL |
| Run | one command (`./dev.sh` or `make dev`) starts API, worker, and frontend; a `Dockerfile` + `docker-compose.yml` with ffmpeg bundled as an alternative |

**Folders**

- `inbox/` watched for new game files (polling every 10 s; ignore files still being written by checking size stability).
- `data/jobs/{job_id}/` holds `probe.json`, `calibration.json`, `timeline.parquet`, `pbp.json`, `events.json`, `cutlist.json`, thumbnails, logs.
- `data/templates/` holds broadcaster templates and reference crops.
- `exports/` holds finished MP4s, cut lists, captions.

**Data model**

- `Job`: id, status (draft, calibrating, ready, analyzing, review, exporting, done, failed), stage, progress 0 to 1, source\_path, sport, game\_id, team, start\_spec, end\_spec, options (JSON), template\_id, error, timestamps.
- `Template`: id, name, sport, broadcaster label, bug box and sub-ROIs (normalized), crop box, reference image path.
- `Clip`: id, job\_id, order, src\_in, src\_out, period, clock, score\_before, score\_after, points, kind (field goal, free throws, touchdown, goal, run, etc.), scorer, description, confidence, enabled, pbp\_event\_id, warnings.
- `Export`: id, job\_id, path, title, settings, created\_at.

**API endpoints**

- `GET /api/fs/browse?path=` list files under allowed roots (home folder by default) for the file picker.
- `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}`, `DELETE /api/jobs/{id}` (deletes the job record and artifacts, never the source file).
- `GET /api/games?sport=&date=&team=` game lookup through the adapter.
- `POST /api/jobs/{id}/calibrate` returns detected boxes and a preview frame; `PUT /api/jobs/{id}/calibration` saves edits.
- `POST /api/jobs/{id}/analyze`; `GET /api/jobs/{id}/events` server-sent events for progress.
- `GET /api/jobs/{id}/clips`; `PATCH /api/clips/{id}` (enabled, src\_in, src\_out).
- `POST /api/jobs/{id}/export`; `GET /api/exports/{id}/file`.
- `GET /api/media/{job_id}/source` streams the source with HTTP range support so the browser can preview any clip without extracting it.
- `GET /api/templates`, `DELETE /api/templates/{id}`.

## Web app screens

Five screens, dark theme, desktop-first. The review screen is where Tim spends his time, so it gets the most care.

1. **Jobs list.** Cards for every job: source file name, game, team, status, progress bar, created date. A "New job" button and an "Inbox" badge showing files waiting in `inbox/`.
2. **New job.** File picker (browses local folders through the API), the game setup form from the user flow, and a game search that shows matching games as selectable rows (date, teams, final score).
3. **Calibration.** A frame from the game with the detected score bug outlined and each sub-field labeled with its OCR read (e.g. "SA 104 / NY 99 / 4th / 1:22 / 24"). A live preview of the 9:16 crop beside it. Frame scrubber to check other moments. Drag handles to adjust the box. "Looks right" saves and continues.
4. **Review.**
   - Left: the clip list. Each row shows a thumbnail, game time, score before and after (e.g. "NY 75 to 78"), points, scorer and description from play-by-play, duration, and a confidence badge. Toggle to enable/disable. Rows with warnings are flagged.
   - Right: a 9:16 player that plays the selected clip from the source using the saved crop, with in/out nudge buttons (±0.5 s) and "Play from here through the end" to watch the whole cut in sequence.
   - Top bar: total runtime of enabled clips, clip count, unmatched play-by-play events (expandable list with "jump to game time"), and the Export button.
   - Keyboard: J/K previous/next clip, Space play/pause, X toggle clip, \[ and \] nudge in point, { and } nudge out point.
5. **Export dialog.** Title text, optional bottom caption, toggles (audio crossfade, HR trot), then render with a progress bar. When done: in-app player, "Reveal in folder", copy caption.

Previews play the original file through `/api/media` with the crop applied by CSS (`object-fit` + transform) so review is instant; only export renders with ffmpeg.

## Config, scope, and build plan

**Keys and config.** Only one secret is required: `ANTHROPIC_API_KEY`. All league data sources are free and keyless. Everything lives in `.env` (with a committed `.env.example`):

- `ANTHROPIC_API_KEY`
- `CLAUDE_MODEL` (default `claude-sonnet-5-5`)
- `CLAUDE_BUDGET_PER_JOB_USD` (default 2)
- `OCR_SAMPLE_FPS` (default 2)
- `ALLOWED_ROOTS` (default the user's home folder), `INBOX_DIR`, `DATA_DIR`, `EXPORTS_DIR`

**Out of scope for this build**

- Downloading, scraping, or fetching game footage from any site, and anything touching DRM. The app only processes files Tim provides.
- Auto-posting to TikTok or other platforms. Leave a `publishers/` interface with no implementation so a scheduler integration can be added later.
- Auth, cloud hosting, mobile layout, live (in-progress) games.

**Testing without real footage.** Real broadcasts are not in the repo, so build a **synthetic broadcast generator** (`tools/make_synthetic_game.py`): it renders a 10 to 20 minute 720p video with moving background noise and an ESPN-style score bug whose scores, period, game clock, and shot clock follow a scripted game, including commercial gaps (bug hidden), replays (clock jumps backward), score-change animations covering the score, and free throws. It also writes the ground-truth cut list. End-to-end tests run against it. Play-by-play adapters get tests against recorded JSON fixtures; record real responses for the 2026 Finals Knicks vs Spurs comeback game if the network allows.

**Acceptance criteria**

- [ ] `./dev.sh` (or `make dev`) on a Mac with ffmpeg installed starts everything; README covers setup in under 10 steps.
- [ ] On the synthetic game: every scripted score by the followed team is detected, zero false clips, clip starts within 1.5 s of the scripted possession start, ends within 1 s of target.
- [ ] Replays and commercial gaps never appear in exported clips.
- [ ] Calibration auto-detects the synthetic bug without manual adjustment, and the crop matches the rule in Output format.
- [ ] NBA game lookup returns real games by date and team; play-by-play matching labels clips with scorer and description.
- [ ] Review screen supports toggle, nudge, and sequential playback; edits persist across reloads.
- [ ] Export produces a valid 1080x1920 MP4 that plays in QuickTime and passes `ffprobe` checks.
- [ ] Analysis of a 2.5-hour 1080p file is estimated under 15 minutes on a laptop CPU (benchmark on a long synthetic file).
- [ ] Unit tests pass for timeline cleaning, event detection, boundary rules for all four sports, and crop math.

**Build order**

1. Repo scaffold, `.env.example`, run script, README skeleton.
2. Synthetic broadcast generator plus ground-truth cut list.
3. Probe, frame sampling, calibration (Claude vision + static-region validation), templates.
4. OCR and timeline building with cleaning rules.
5. Score events and NBA clip boundary rules; tests against synthetic ground truth.
6. NBA adapter, game lookup, play-by-play matching, auto start point.
7. Export renderer (crop, canvas, title, crossfades, cut list, caption).
8. Frontend screens and API wiring.
9. NFL, NHL, MLB adapters with fixture tests.
10. Docker option, README polish.
11. Final self-review: run the full test suite and one full synthetic job through the UI flow, then write `BUILD_NOTES.md` covering what was built, decisions and assumptions made, known gaps, and exactly what Tim should test first with a real game file.
