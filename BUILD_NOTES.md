# Build notes

What was built, what was verified and how, every decision taken without asking, what is
still open, and what to test first. Written at the end of the build (2026-10-05).

## Where it stands

All eleven steps of the PRD's build order are in the repo, plus what was added after the
first real run: uploading a game from the browser, a camera check that trims crowd shots
off clip edges, progress steps with a time estimate, a hosted copy on Railway behind a
password, ESPN as a second NBA data source, and a "defensive plays" option.

| | State |
| --- | --- |
| NBA pipeline (probe, calibrate, sample, timeline, clips, match, export) | Built. Verified on synthetic games against ground truth and on one real broadcast. |
| Web app (upload, setup, calibrate, review, export) | Built. Driven end to end in a browser on a synthetic game and on the real game. |
| Claude (vision calibration, re-reading unclear frames, captions) | Built and run live on the real game. About 25 cents for the whole job. |
| NFL, NHL, MLB adapters | Built and unit tested against recorded league data. Never run on real footage. |
| Hosted on Railway | Live at <https://possession-cut-production.up.railway.app>. The image built first time and passes the pipeline self-check there on every deploy. It is closed until you set a password, and no game has been run on it yet (see "Hosted on Railway"). |
| Docker | The image builds and runs on Railway. `docker compose` on a local machine is unrun: Docker is not installed here. |
| The Vercel address | Forwards to the Railway app, checked in a browser. (It first pointed at an engine on your own PC, which is what the "Start the engine" screen was.) |
| Auto-start at login | Script written. Not run by me: it changes what starts when you log in, so that click is yours. |

Tests: 343 pass. 299 are logic tests that run in seconds; 44 render a synthetic broadcast and
run the pipeline and the API on it. Lint is clean and the frontend type-checks and builds.

## What to test first with a real NBA game

1. In Railway, open the project **possession-cut**, the service, **Variables**, and add
   `APP_PASSWORD` (a password you choose) and `ANTHROPIC_API_KEY` (your Claude key).
2. Open <https://possession-cut-production.up.railway.app> (the Vercel address forwards
   there) and sign in.
   To run it on this PC instead: double-click `start.cmd` and open <http://127.0.0.1:8000>.
3. Click **Upload a game video** and pick a full broadcast. On Railway that goes over your
   internet connection (roughly 20 to 70 minutes for 5 GB); on this PC, about a minute.
4. Game setup: set the date, **Find games**, pick the game, pick the team, leave the start
   on **Auto: start of the biggest run**, save.
5. Calibration: check that each field reads what the frame shows (scores, period, clock,
   shot clock). Click through "Other moments in the file". If a box is off, drag it. Confirm.
6. Wait for the analysis (5 to 10 minutes). The screen shows the step it is on and the
   time left.
7. Review: play the whole cut once with **Play from here through the end**. Look for the
   things listed under "What to check by eye" below.
8. Export and download the video. (On this PC you can also run
   `python tools/verify_export.py exports/<file>.mp4 --calibration data/jobs/<id>/calibration.json`.)

Try a second broadcaster early (TNT, NBC, Prime). Everything below was tuned on one ESPN/ABC game.

## The real-game test (2026 Finals Game 4, Spurs at Knicks)

**The file.** `NBA_20260610_SAS_NYK_1080p60_ABC_mkv.mp4` in Downloads: 5.17 GB, 1280x720
at 59.94 fps, 2:25:32. The note said the name starts with `NBA_20250610`; it starts with
`NBA_20260610`. A frame confirmed the game before anything ran: the bug reads SA / NY with
the Finals logo and "NY LEADS 2-1", at Madison Square Garden on ABC.

**The run, through the app's own screens.**

| Step | Result |
| --- | --- |
| Upload (the page's protocol, sent by script) | 71 s, byte-identical copy. An interrupted upload resumed from 50 MB after an engine restart. |
| Game lookup, date 2026-06-10 | "San Antonio Spurs at New York Knicks, SAS 106, NYK 107, NBA Finals, Game 4" |
| Auto start | "Down 29 (52-81) at Q3 9:40", after D. Fox's jumper |
| Calibration | Claude vision, confidence 100%, all seven fields read correctly, no adjustment |
| Analysis | About 8 minutes. 17,465 samples; Claude re-read the 227 unclear ones |
| Play-by-play | 109 score changes matched, none unmatched either way; final read as 106-107 |
| Cut | 22 clips, 6:14, every Knicks point from 52 to 107 |
| Export | 1080x1920, H.264 High, 59.94 fps, 436 MB. `verify_export.py`: 19 of 19 |

**What the first run got wrong, and what changed.** The first cut had five clips of about
four seconds, an and-one read as free throws, and a 42 second clip made of two possessions.

1. **The bug is slower than the PRD assumed.** ESPN's bug shows a score 2.2 to 3.8 s after
   the make (median 2.75 s over 68 baskets); the PRD says 0.5 to 2 s. The shot clock resets
   when the ball drops, so the make's own reset fell outside the guard and was taken for
   the start of the possession. Now the last reset before a score is recognised as the make
   itself, and only signals before it can start a possession.
2. **Clip ends follow the make.** "Score + 1.5 s" would have ended clips 4 to 5 s after the
   ball dropped. A clip now ends 2.5 s after the make, but not before the new score has
   been on screen for half a second, and never later than the PRD's score + 1.5 s. At the
   lag the PRD assumed, this gives the PRD's own answer.
3. **Free throws are placed by the measured lag.** The bug's lag is measured from the
   game's baskets and each free throw window runs from 2 s before the ball drops to 2 s
   after. With the old fixed window the clip began with the ball already in the air.
4. **Resets shown as 23.** The graphic sometimes skips 24 and first shows 23. Ten
   possession changes were being missed that way; one of them caused the 42 second clip.
   A reset the operator corrects within a moment (24, then 14) now counts too.
5. **Baskets through a foul.** The clock stops on the whistle, and by the time a slow bug
   shows the basket it has been stopped for three seconds, which looked like free throws.
   The test is now how long the clock had been stopped beforehand (27 s at the least for a
   real free throw in this game; 3 to 4 s for an and-one). When play-by-play is matched it
   settles the question.
6. **The bug returning mid-possession.** Two possessions began while the bug was off
   screen. Those clips now start where the bug returns and are no longer flagged as
   over-long possessions.
7. **Crowd shots and close-ups.** One clip opened on five seconds of crowd and a player's
   face while the clock ran; others ended on a close-up of the scorer. The bug cannot see
   that, so a camera check was added (see "Camera check" below). It trimmed 6 of 20 clips,
   14 seconds in all.
8. **Calibration on a real bug.** ESPN's period, clock and shot clock sit so close that
   text detection returns them fused ("2ND3:0020"). The local detector now splits those
   and finds field edges from the pixels. It reaches 100% on this file without Claude.
9. **The hosted page could not have connected.** The engine answered Chrome's
   local-network permission check with an error. Fixed and tested.

**How the cut was reviewed.** I looked at every clip of the cut as one frame per second,
plus the first and last second at four frames per second, before the camera check was
added. After adding it I confirmed its six trims against those frames and looked at four
frames of the export. I did not watch the cut at speed and did not listen to it.

**What to check by eye.**

- **Clips 7 and 8** start a few seconds into the possession, because the bug was off screen
  when it began.
- **Clips 12 and 21** are long (24.5 s and 28.5 s). The shot clock says they are full
  possessions ending in late shots. Trim the front in review if they drag.
- **Clip 17** opens with four seconds from a high, wide camera. It is live play, so it stays.
- **Clip ends.** Most end about three seconds after the make with the new score just
  showing. Clips 6, 8 and 17 end slightly earlier, before a close-up, so the score has not
  ticked over yet.
- **Free throws** (clips 1 and 15, and the tail of clip 6) are about 5 s per make. The
  second free throw of clip 1 is shown from a tight camera; it is the shot itself, so it stays.
- **Audio at the cuts.** Crossfades are verified by measurement on the synthetic game only.
- **The caption bar** says "NYK leads 3-1", the series after this game. The bug in the
  picture says 2-1, the series before it.

## The second real game (2016 Finals Game 7, Cavs at Warriors, on the hosted copy)

You ran this one yourself on Railway on 2026-10-06 and reported a poor cut, a lagging
player, and only Cavs clips although "Opponent scores" was on. I could not open your job
(I do not sign in to the hosted copy), so I read the server's logs and ran the same file,
ESPN's 720p "FULL GAME" upload from your Downloads folder (1.2 GB, 1:47:34, 1.5 Mbit/s),
through the pipeline here exactly as the server would: no Claude, fresh detection.

**What went wrong, in order.**

1. **The game lookup failed.** You searched 2026-06-19 (the game is 2016-06-19), and at
   that moment the server was still running the build from before the ESPN fallback, so
   the lookup waited 50 s on NBA.com and answered "unavailable". The job went on without a
   game: no play-by-play, no scorer names, no cross-check. The setup page now notices a
   year in the file name that differs from the date ("The file name mentions 2016") and
   offers to switch, and the hosted copy no longer tries NBA.com at all.
2. **The Warriors' score box read as nothing.** The built-in detector put the home score
   box one row too tall, so it took in the bar's white top line. The recognizer then saw a
   long bar with two small digits under it and returned an empty string, in every frame.
   No Warriors score changes means no Warriors clips; "Opponent scores" had nothing to
   add. The clock box ran into the shot clock beside it the same way and read "5:5716".
   The calibration screen did warn ("Needs a manual check", scores read in under 70%),
   but the boxes looked right at a glance, and nothing said which one was broken.
3. **The player lagged** because the review screen streamed the original file from
   Railway: 896 range requests in nineteen minutes, each a second or so, and a keyframe
   only every six seconds in that file, so every jump to a clip waited for the next
   keyframe to come over the internet.

**What changed.**

- Field reads drop line-like ink before recognizing (a border, timeout dashes, an
  underline), box padding stops at a line above or below the text as it already did at
  artwork beside it, and a box snapped along the row may not reach into the next field's
  text. On top of that, calibration now pulls in any box that still reads poorly, trying
  a few smaller boxes and keeping the smallest that reads cleanly (scores must also never
  go down across the sampled frames). The 2016 file now calibrates at confidence 1.0
  with every field read in every frame, against 0.0 before.
- The analysis names the side when one score box is the problem: "GS's score could not
  be read in 93% of live samples, so its baskets are mostly missing. Fix the home score
  box in calibration and analyze again."
- A hosted copy makes a small preview copy of every upload in the background (480 tall,
  a keyframe every second, ~0.7 Mbit/s) from the moment the file is probed, and the
  review screen plays that. Until it is ready the original plays and the page says so;
  when it lands the player switches to it on the selected clip. On your own computer
  nothing changes: a playable file plays as it is, as before.

**The cut, with the game attached and both teams.** 94 score changes found, all 94
matched to the play-by-play (93 by the clock), none unmatched; two early plays in the
play-by-play have no counterpart in the video, which is normal for ESPN's "full game"
uploads, which trim dead time. Clock readable in 98.6% of live samples and both scores in
97.2%. 76 clips: 41 Cavs, 35 Warriors, every one with a scorer. Before the fixes the same
file gave 72 clips with 22 phantom Warriors plays, a clock readable 7% of the time, and
free throws of one point everywhere.

**What to do on the hosted copy.** Deploying this build wipes the uploaded file, so
upload it again, search the game with the date 2016-06-19, pick CLE, turn on "Opponent
scores", and let calibration run. Setting `ANTHROPIC_API_KEY` on the Railway service
(Variables) is still worth doing: Claude vision reads unusual layouts better than the
built-in detector and writes the captions. The cut you reviewed was made without it.

## Upgrades round (2026-10-06)

Asked for: a control to nudge every clip at once, "and any other changes". The every-clip
control was built first (In and Out ±0.5 for the whole cut, Alt + brackets on the keyboard,
"Reset all edges"). For the rest I had five reviewers each read one part of the app
(review screen, setup and calibration, export, jobs list and upload, API and worker) and
propose concrete upgrades with evidence from the code; the verification pass was cut
short by a usage limit, so I picked from the proposals myself, favouring what a highlight
editor would notice over what a developer would like. Not built, on purpose or for later:
duplicating a job for the other team (the "Opponent scores" toggle covers most of it),
re-attaching a fresh upload to a job after a server restart (worth doing; the biggest
hosted annoyance left), a per-clip lower third drawn in the bottom bar (a day's work and
a PRD change), splitting a long cut into parts, a worker watchdog, a plain-words failure
mapping, and phone-width layouts.

Built, in four commits:

- **Review:** filter chips and a least-confident-first sort, "turn these N off / on",
  undo and redo (Ctrl+Z) for every clip change through one bulk route, I and O to set an
  edge at the playhead with frame and second stepping, make markers on the scrubber with
  the reason behind each confidence badge, playback speed, cut-wide progress in the
  sequence pass, and reopening on the clip left last time.
- **Worker:** edits survive a re-analysis (matched by play-by-play event, else by period,
  clock, kind and score), cancel stops calibration too and drops a cancelled export, a
  missing game file fails at once in plain words, the jobs list shows the next step and
  the newest video with Download, deleting a job can take its videos (the default on a
  server), and the hosted sweep removes videos whose job is gone.
- **Export:** sound choice (broadcast, levelled with loudnorm at -14 LUFS once over the
  whole cut, or silent), chapters in the MP4, and a YouTube timestamp block in the
  caption file and the export result.
- **Setup and calibration:** per-field read counts across the sampled frames with a jump
  to the first failing frame and a hint about why, the bug's team labels against the
  followed team with a warning when neither matches the game picked, and form checks for
  a clock past the period, an end before the start, and an end before the biggest run.

## Things I did differently from what was asked

You asked to be told afterwards.

- **Upload was added after you saw the first version.** The PRD says files are registered
  by path and nothing is uploaded. Both now exist: the Upload box is the front door, and
  "pick it from disk" and `inbox/` remain for files already on the machine.
- **There are now two ways to run it.** On your computer (the engine started by
  `start.cmd` or the autostart, opened through the Vercel page or `127.0.0.1:8000`), and
  on Railway, which you asked for after seeing the first. The PRD describes only the first.
- **A password and server restrictions were added** for the hosted copy; none of that is
  in the PRD.
- **A self-check command** (`python -m possession_cut.selfcheck`) runs the pipeline on a
  scripted clip. Railway runs it before each deploy.
- **Clip ends and free throw windows follow the make**, not the score (items 2 and 3 above).
- **Play-by-play decides free throw versus basket when matched.** You said play-by-play
  only labels and cross-checks. Timing still comes only from the bug; this is the
  cross-check overriding a guess the bug cannot make reliably.
- **Camera check**: new, on by default, not in the PRD.
- **ffmpeg was installed** with `winget install Gyan.FFmpeg`. The PRD assumes it is there.
- **Windows launchers** (`start.cmd`, `dev.cmd`, `dev.ps1`, autostart) beside the Mac ones.
- **`opencv-python`, not `-headless`**: the OCR package depends on it and the two conflict.
- **Game lookup uses `scoreboardv3`** directly; `nba_api`'s scoreboard returned half-empty
  rows for the 2026 Finals.
- **Game lookup and the calibration read-out run in the API**, not the worker, so they
  answer while a long analysis is running.
- **Export uses seeked inputs and the concat filter**, not the concat demuxer, which cannot
  cut on exact frames or crossfade audio.
- **Title and caption are drawn with Pillow**, not ffmpeg `drawtext`, so any text works.
- **The local bug detector is a full route**, not a fallback stub, because Claude could not
  be reached for most of the build.
- **Two start signals beyond the PRD's three**: the team's own previous score, and the bug
  returning from a break.

## Decision log

### Environment

- **This machine is Windows 11, the PRD says Mac.** Everything is cross-platform. `dev.sh`
  and `make dev` cover Mac and Linux; the `.cmd` and `.ps1` files cover Windows. All call
  the same launcher (`python -m possession_cut.dev`).
- **Python 3.12 venv** at `.venv` (3.13 is the system default; the PRD pins 3.12).
- **The API key and workspace ID live only in `.env`**, which is gitignored.
- **The repo sits inside OneDrive.** Uploads therefore go outside it (see Upload). `data/`,
  `inbox/` and `exports/` still default to the repo folder; move them with `DATA_DIR`,
  `INBOX_DIR` and `EXPORTS_DIR` if syncing them becomes a nuisance.

### Sampling and OCR

- **Sample timestamps are frame-exact.** ffmpeg's `fps` filter, with default rounding,
  returns a frame about a quarter second later than the slot it labels. The sampler uses
  `round=up`, and a test decodes a per-frame barcode to prove sample k is the frame at k/fps.
- **Digit-restricted OCR is done at the probability level.** The recognizer's output is
  masked to the characters a field may contain before decoding. The models are the
  PP-OCRv4 ONNX files that ship inside `rapidocr-onnxruntime`; nothing is downloaded.
- **Field crops are trimmed to the ink before recognition.** On the synthetic game this
  took misreads from 1 in 700 to 0 in 3,500.
- **A field is re-read only when its pixels change**, so most samples cost one or two OCR calls.
- **Crops are taken on even pixel boundaries.** ffmpeg silently rounds odd crop offsets on
  4:2:0 video. A benchmark at 1080p exposed this; a test covers it.

### Calibration

- **Three routes, one validation**: a saved template that matches by image, Claude vision,
  and a fully local detector. All end in the same checks: the bug region must be static
  across frames, and the fields must parse as scores and a clock.
- **Claude's boxes are never used raw.** The bug box is snapped to the static region and
  each field to the text actually there. Claude says which text is which; local detection
  supplies the pixels.
- **Field edges come from ink, not from text boxes.** Along the bug's main row, a column
  that never has ink in any frame is a gap between fields. That is what separates
  "7:52" from "13" when the detector fuses them.
- **Score boxes are padded for a third digit, but padding stops at artwork** such as a logo,
  and (since the 2016 Finals test) at a line above or below the text, such as the bug's
  border or a row of timeout dashes.
- **Line-like ink is dropped before recognizing.** A border or an underline that gets into
  a box made the recognizer return nothing for the whole field; now wide flat shapes and
  thin bars on a box's edge are blanked first. A colon's dots and a "1" are not affected.
- **A box may not reach into its neighbour's text.** Ink runs fuse a game clock and the
  shot clock when the two panels touch; the clock box stops where the shot clock's text begins.
- **Boxes that still read poorly are pulled in automatically**: a few smaller boxes are
  tried on the sampled frames and the smallest that reads cleanly wins. Scores must not go
  down across frames, so a box that clips "129" to "29" does not win. Manual boxes are
  never touched.
- **Bug visibility is image similarity against the template**, judged on the better half
  of six slices, so a score animation over one team does not count as "bug hidden".
- **Crop guard beyond the PRD rule.** The rule (full height, width = bug width / 0.78,
  centred, clamped) is applied as written. A small corner bug would give a crop narrower
  than it is tall, so that case falls back to the reference 1.18:1 shape.

### Claude

- The key supplied is not scoped to a workspace, so requests carry
  `ANTHROPIC_WORKSPACE_ID`. With it set, all three uses ran live on the real game:
  vision calibration (8 cents), re-reading 227 unclear samples on 29 contact sheets, and
  the caption. The per-job budget is 2 dollars (`CLAUDE_BUDGET_PER_JOB_USD`).
- Requests use `CLAUDE_MODEL` (default `claude-sonnet-5-5`) with structured outputs and the
  server-side fallback beta, dropping to a plain request if the account does not accept it.
- The re-read declines to run when over a quarter of samples read poorly; that is a
  calibration problem, not something to pay Claude to paper over.

### Timeline, events and clip boundaries

- **Cleaning is global, not greedy.** "Scores never decrease, the clock only runs down" is
  enforced by keeping the longest consistent chain of reads, so one confident misread
  cannot poison what follows.
- **Replays that re-air the bug** are caught by the clock sitting behind the game, or the
  score sitting below one that already held.
- **Possession start** = the latest of these that happened before the make, less 1 s:
  1. the shot clock jumping to 24 or 14, taken when it starts counting down again;
  2. the game clock starting after a stoppage of 2.4 s or more;
  3. the opponent's score appearing (no pre-roll, so their make is not shown);
  4. the followed team's own previous score;
  5. the bug returning from a break of 4 s or more;
  plus the shot clock switching off in the last 24 s of a period.
- **The make** is the last shot clock reset within 4.5 s before the score (8 s if the shot
  clock never ran in between). With the shot clock off, the game clock stopping does the
  same job: it stops on a make in the last minutes.
- **And-one:** a free throw by the same team, next score in the game, out of the same
  stoppage as the basket, rides on the basket clip as a trailing segment. A slow bug can
  show the last free throw after play has resumed, so stoppages are compared by the clock
  value they held at, not by the clock when the score appears.
- **Clips have segments.** `Clip.segments` holds the kept ranges; `src_in` and `src_out`
  are the outer bounds.
- **Not-live cut-outs are conservative**: widened to the neighbouring live samples, so no
  frame of a replay or commercial can leak in.
- **A score first seen after a break** ends its clip at the cutaway rather than after it.
- **Shot clock reads are validated against physics**: it counts down in real time, holds,
  or jumps to a reset value. A reset must land on 24, 14, or one below with consistent
  reads after it.

### Defensive plays (added on request, 2026-10-06)

- **Blocks and steals** by the followed team become 7 s clips (3 s before the play to 4 s
  after) when "Defensive plays" is on. They come from play-by-play: cdn.nba.com has them
  as their own actions, stats.nba.com says "X STEAL" in a description with a blank type,
  and ESPN writes them inside the shot or turnover ("... turnover (Josh Hart steals)",
  "Mitchell Robinson blocks ..."). All three agree on the Finals game: 8 blocks, 16 steals.
- **Video time** for each comes from the game clock through the bug's timeline, the same
  way game-time start points do, so it is good to about half a second.
- **A steal that leads straight to a score joins that score's clip** (the clip then starts
  3 s before the steal and is labelled by the score). One with no score after it stands
  alone with "STL" or "BLK" and no points.
- Not available for a synthetic game's sidecar play-by-play, and untested on real footage
  through the app: the clip builder is tested on the scripted game and the parsers on the
  recorded Finals data.

### ESPN as a second NBA source (2026-10-06)

- NBA.com refuses cloud servers: from Railway stats.nba.com stalls and cdn.nba.com answers
  403 (checked from the server itself). The lookup now tries stats.nba.com once with an
  8 s timeout, then uses ESPN's scoreboard; a game found there (id `espn:<event>`) takes
  play-by-play from ESPN's summary. ESPN's team codes (SA, NY, GS, NO, UTAH, WSH) are
  translated to the NBA's. After one failure NBA.com is left alone for ten minutes.
- ESPN's 109 scoring plays for the Finals game agree with NBA.com's play for play, and
  the auto start point comes out the same. Verified from Railway: the lookup returns the
  game and its 109 plays.

### Camera check

- The bug says play is live; it does not say what the director shows. `pipeline/camera.py`
  samples small frames of each scoring clip twice a second, learns which colour is the
  playing surface (the commonest colour across the clips: hardwood here), and calls a
  frame a cutaway when it shows under 15% of the usual amount of it. On the real game,
  game-camera frames showed 65% to 170% of the usual amount and cutaways under 5%.
- It trims a cutaway that begins within 2.5 s of a clip's start, and one that runs to the
  clip's end (at most 2 s). It never touches the middle of a clip, a free throw window,
  or a clip that would drop under the minimum length. A possession shown entirely from
  another camera is left alone.
- "Game camera only" in game setup turns it off.

### NBA data, matching, start point

- **Play-by-play sources, in order:** cdn.nba.com liveData (recent seasons), stats.nba.com
  `playbyplayv3` (all seasons), then `nba_api`. Cached per game under `data/cache/nba/`.
- **Points are derived from the running score**, so both feeds parse identically.
- **Recorded fixtures** for 2026 Finals Game 4 (game ID `0042500404`) are in
  `backend/tests/fixtures/nba/`.
- **Matching runs three passes, strictest first**: running score + period + clock within
  3 s; the PRD rule alone; then running score with the clock up to 30 s off.
- **Auto start** = the last opponent score that put the team down by its largest deficit.
- **Sidecar play-by-play:** `<video>.pbp.json` beside the source is used instead of the
  league API. That is how the synthetic game gets labels.

### Export

- **One ffmpeg run, one encode.** Each segment is its own seeked input, trimmed in a filter
  graph, concatenated, cropped, scaled to 1080 wide and padded to 1080x1920.
- **Frame-exact.** Seeks are biased half a frame early and trims sit half a frame before
  the wanted frame. Verified: every exported frame's barcode equals the planned source frame.
- **Audio crossfade is two whole frames wide** (66.7 ms at 30 fps), inside the PRD's 60 to
  100 ms, and audio and video stay the same length through any number of cuts.
- **More than 48 segments render in batches**, joined with video stream copy and a single
  AAC encode, so there are no audio seams.
- **Sources a browser cannot play** (MKV, TS, AC-3, HEVC) get a preview proxy for review.
  Export always cuts from the original.
- **Output frame rate is the source's**, up to 60. The real export is 59.94 fps.

### Upload

- The page sends the file to the engine in 8 MB pieces (`/api/uploads`). The engine
  appends them in order and can say how much it has, so a dropped connection or an engine
  restart resumes instead of starting again.
- **Uploads are kept outside cloud-synced folders.** When `data/` sits inside OneDrive,
  Dropbox or iCloud, uploads go to the computer's local app-data folder. On this machine
  `.env` sets `UPLOADS_DIR` to `Videos\Possession Cut\uploads`.
- **An uploaded file belongs to the app** and is deleted with its job, unless another job
  uses it. A file picked from disk or dropped in `inbox/` is never deleted.
- **Windows locks.** Antivirus and the search indexer briefly lock a file that is being
  written; the first real upload hit this after six pieces. The engine waits up to six
  seconds for the lock to clear, then asks the page to retry.
- Refused before any data is sent: a file that is not a video, and a disk without room.

### App

- **Job queue is the jobs table.** The worker claims one task at a time and writes
  progress back to the row. A worker that dies mid-task requeues it on restart.
- **Re-analysis is cheap.** Bug reads are cached against the calibration, so changing the
  team, start point or toggles and re-running takes about 30 seconds on a full game.
- **Progress steps and the time estimate** are worked out in the API from what the worker
  already records. The estimate is time so far scaled by how much of the bar is left.
- **The connect screen keeps looking** for the engine and opens by itself once it is up.
- **One engine at a time.** Starting a second copy exits quietly.
- **No database migrations.** The schema is created on first run. Nothing added since the
  first version needed a new column.

### Benchmark

`tools/benchmark_analysis.py --minutes 150 --height 1080` on this machine (i5-12600K, 6
workers): 147.5 minutes of 1080p analyzed in 7.1 minutes (21x real time). The real 720p60
game sampled at 17 to 29x real time depending on what else was running. The PRD target is
under 15 minutes.

## Known gaps

- **`install-autostart.cmd` has not been run.** In particular I have not seen whether any
  window flashes at login.
- **One real game.** The lag handling, the 23-reset rule and the free throw test are tuned
  on ESPN/ABC's bug.
- **Replays that keep the live bug on screen.** During a dead ball some broadcasts play a
  replay under the live bug. The bug cannot show that, and the camera check only looks at
  clip edges and only at colour, so a replay from the main camera angle would pass. None
  turned up in this cut.
- **The tail of a replay that re-airs the bug** can match the live state exactly for a
  second or two.
- **The camera check depends on a surface colour.** It should carry to football, hockey
  and baseball wide shots, but a close-up with grass behind the player would not be
  caught. Tested on one real NBA game and synthetic footage.
- **NFL, NHL, MLB** have never seen real footage. Known holes: the calibration read-out
  and the Claude re-read parse fields generically, so MLB innings and NFL ":05" clocks
  show as unreadable there; game-time start points need a clock, so they do not work for
  MLB; MLB halves must OCR for bottom-half runs to match; an NHL shootout winner has no
  play-by-play event; overtime lengths are not game-specific. NFL drive mode is not built.
- **No real game has been run on Railway.** I did not sign in to the hosted copy (the
  password is yours to set), so the upload over the internet, the analysis speed on
  Railway's processors and playback from the server are untried there. What is proven on
  the server itself: the image builds, the app starts, and the self-check (render, find
  the bug, cut, export) passes.
- **On Railway an uploaded game does not survive a restart or redeploy** (see "Hosted on
  Railway"). There is no way yet to re-attach a fresh upload to an existing job: after a
  wipe, the game goes up again as a new job.
- **One password, one user.** No accounts, and jobs run one at a time.
- **Anamorphic and interlaced sources** are handled in the export graph but untested on
  real files.
- **The time estimate is simple.** It is steady during the long stretches and jumpy in
  the first seconds of a task.
- **A local engine has no login** unless `APP_PASSWORD` is set. It listens on this
  computer only, which is what makes that acceptable.
- **Publishing** (TikTok and the rest) is an interface only, as the PRD scopes it.

## Hosted on Railway

Project `possession-cut` in your Railway workspace, one service of the same name, built
from the repo's `Dockerfile`. Address:
<https://possession-cut-production.up.railway.app>.

**To open it, two variables are yours to set** (Railway dashboard, the service, Variables):

| Variable | Value |
| --- | --- |
| `APP_PASSWORD` | A password you choose. This is the sign-in. Until it exists the app answers nothing. |
| `ANTHROPIC_API_KEY` | The same key that is in `.env` here. Optional: without it the built-in detector and a template caption are used. |

I did not set these because they are your secrets. Railway restarts the service by
itself when a variable changes. Everything else is already set (`HOSTED`, the folders,
the clean-up days and `ANTHROPIC_WORKSPACE_ID`).

**How it is laid out**

| Where | What | Survives a redeploy |
| --- | --- | --- |
| Volume at `/data` (5 GB, the Hobby plan's limit) | Job records, saved bug layouts, play-by-play cache, finished videos | Yes |
| The container's own disk, `/scratch` (up to 100 GB) | Uploaded games, preview copies, export working files | No |

A game is bigger than the volume, so it lives on the temporary disk. That has one
consequence worth remembering: **a redeploy or restart wipes uploaded games.** The job
and its clip list stay, but review playback and export need the file, so the game would
have to be uploaded again. Finish and download a cut before new code is deployed. On
the Pro plan (50 GB volume) set
`UPLOADS_DIR=/data/uploads` and this goes away.

**What keeps the disk from filling.** Uploaded games are deleted 7 days after their job
last changed and exports after 30 (`UPLOAD_RETENTION_DAYS`, `EXPORT_RETENTION_DAYS`).
Before an export starts, the oldest finished exports are removed if the volume lacks
room. Download what you want to keep.

**What the password protects.** Every API route except the health check needs the
session cookie that signing in sets (HttpOnly, 30 days, signed with a key derived from
the password, so changing the password signs everyone out). Eight wrong guesses from one
address lock that address out for ten minutes. On a hosted copy the page has no view of
the server's disk: no file picker, no job from a path, no "reveal in folder".

**The self-check.** `python -m possession_cut.selfcheck` renders two minutes of a
scripted game and runs calibration, analysis and export on it, checking each against the
script. Railway runs it before every deploy (the service's pre-deploy command), so a build with a broken
ffmpeg, OCR model or font is refused instead of going live. It took 32 seconds on
Railway and passed every check there, against about two minutes on this PC.

**Deploying new code.** My pushes to `main` did not start a deploy by themselves; each
deploy so far was started by re-attaching the repository. That means the Railway GitHub
App most likely has no access to this repository for push notifications. Until it does
(GitHub, Settings, Applications, Railway, Configure), deploy with Ctrl+K, **Deploy
Latest Commit** in Railway. The service only redeploys for changes under `backend/`,
`frontend/`, `tools/` or to the `Dockerfile`, so a docs-only push never wipes an
uploaded game. `/api/health` shows which commit is live as `build`.

**Service settings**, set on the service because Railway has deprecated `railway.json`:
build from `Dockerfile`, pre-deploy command `python -m possession_cut.selfcheck`, health
check `/api/health`, restart on failure.

**Cost.** The Hobby plan is 5 dollars a month including 5 dollars of usage, billed at
20 dollars per processor core per month and 10 per GB of memory per month for what is
actually used. The app idles on a fraction of a core and under 1 GB. My estimate for
light use is 5 to 10 dollars a month; I have not measured it. Do not turn on Railway's
"sleep when idle" setting: sleeping stops the container, which wipes uploaded games.

**What changes compared with running it at home.** Uploads go over your internet
connection: roughly 20 to 70 minutes for 5 GB on typical home upload speeds, against
about a minute locally. Analysis ran at about 70x real time on the 2016 game (1:47 of
video in 90 s).

**The review player on a server plays a preview copy**, not the upload: 480 tall with a
keyframe every second, made in the background by the worker from the moment the file is
probed, with three encoder threads so a calibration or analysis alongside it is barely
slowed. A full game takes some minutes to encode; until it is ready the page plays the
original and says so, then switches. The copy sits on the temporary disk with the upload
and goes with it. If an encode fails, the page says why and the original keeps playing.

## Vercel

The Vercel project `possession-cut-pro` (root directory `frontend`) now only forwards to
Railway. `frontend/vercel.json`:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "redirects": [
    { "source": "/", "destination": "https://possession-cut-production.up.railway.app/", "permanent": false },
    { "source": "/(.*)", "destination": "https://possession-cut-production.up.railway.app/$1", "permanent": false }
  ]
}
```

It began as the frontend alone, talking to an engine on your own computer, and showed
"Start the engine to begin" to anyone without one. My first redirect missed the front
page (`/:path*` does not match `/` on Vercel), which loaded while its files were sent
elsewhere and came up blank for a few minutes; the front page now has its own rule.

Vercel had suggested a two-service setup and asked for three things to be confirmed:

- **Service names:** one Vercel project, which is only a redirect. The app is the Railway service.
- **What is public:** the redirect, and the Railway app's sign-in screen. Everything else needs the password.
- **Bindings:** none.

The redirects are temporary (307) so the address can be pointed somewhere else, or back
at a static frontend, without browsers remembering the old answer.
