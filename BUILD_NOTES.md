# Build notes

Running log of what was built, decisions taken without asking, and known gaps.
Finalized at the end of the build; sections are filled in as each step lands.

## Decision log

### Environment (step 1)

- **This machine is Windows 11, the PRD says Mac.** Everything is built cross-platform.
  `dev.sh` / `make dev` cover Mac and Linux; `dev.cmd` / `dev.ps1` cover Windows. All four
  call the same Python launcher (`python -m possession_cut.dev`).
- **ffmpeg was not installed.** Installed FFmpeg 9.0.2 with `winget install Gyan.FFmpeg`
  (user scope). A shell opened before the install will not have it on PATH, so the app also
  looks in the WinGet links folder. `FFMPEG_PATH` / `FFPROBE_PATH` in `.env` override.
- **Python 3.12 venv** at `.venv` (3.13 is the system default; the PRD pins 3.12).
- **The API key lives only in `.env`**, which is gitignored. It is never committed.
- **OpenCV is the standard `opencv-python` wheel, not `-headless`.** `rapidocr-onnxruntime`
  depends on it directly and the two wheels conflict when both are installed. No GUI
  features are used. The Docker image installs `libgl1` for it.
- **The repo sits inside OneDrive.** `data/`, `inbox/` and `exports/` default to the repo
  folder, which means OneDrive will try to sync multi-GB files. Point `DATA_DIR`,
  `INBOX_DIR` and `EXPORTS_DIR` at a folder outside OneDrive in `.env` if that bites.

### Sampling and OCR (steps 3 and 4)

- **Sample timestamps are frame-exact.** ffmpeg's `fps` filter, with its default rounding,
  returns a frame about a quarter second later than the slot it labels. The sampler uses
  `round=up`, and a test decodes the per-frame barcode to prove sample k is the frame at
  k/fps. Without this every boundary would have been 0.23 s late.
- **Digit-restricted OCR is done at the probability level.** The recognizer's per-step
  output is masked to the characters a field may contain (digits for scores, digits plus
  `:` and `.` for clocks) before CTC decoding, rather than filtering text afterwards.
  The models are the PP-OCRv4 ONNX files that ship inside `rapidocr-onnxruntime`, so
  nothing is downloaded at runtime.
- **Field crops are trimmed to the ink before recognition.** A "7" in a box sized for
  three digits otherwise reads unreliably. On the synthetic game this took misreads from
  1 in 700 to 0 in 3,500.
- **A field is re-read only when its pixels change** (small signature compare). The clock
  changes once a second and scores rarely, so most samples cost one or two OCR calls.

### Calibration (step 3)

- **Three routes, one validation.** (1) A saved template that matches by image similarity,
  (2) Claude vision for where the bug is and which field is which, (3) a fully local
  detector. All three end in the same checks: the bug region must be static across frames,
  and the fields must actually parse as scores and a clock.
- **Claude's boxes are never used raw.** Vision models place boxes approximately. The bug
  box is snapped to the static region and each field box to locally detected text, which
  is what makes sub-ROIs tight enough for OCR. Claude contributes semantics (which text is
  which), local detection contributes pixels.
- **The local detector is a full route, not a stub**, because the Claude route could not be
  run live (see "Claude API key" below). It finds the recurring game clock, grows the
  static region around it into the bug box, zooms in, and assigns fields by what they
  contain (clock pattern, period words, 2-4 letter labels, numbers next to labels, the
  small number next to the clock).
- **Score boxes are padded for a third digit, but padding stops at artwork.** A team logo
  beside a score otherwise reads as an extra "1".
- **Bug visibility is image similarity against the template's reference crop**, judged on
  the better half of six slices so a score animation covering one team block does not
  count as "bug hidden". Commercials with numbers in the same screen position do not match.
- **Crop guard beyond the PRD rule.** The rule (full height, width = bug width / 0.78,
  centred on the bug, clamped) is applied as written. One addition: a small corner bug
  would yield a crop narrower than it is tall, so in that case the crop falls back to the
  reference 1.18:1 shape, placed to keep the whole bug inside it. Editable in calibration.

### Claude API key

- The key supplied for this build authenticates but is **not scoped to a workspace**. The
  Messages API rejects it without an `anthropic-workspace-id` header, and the key cannot
  list workspaces. `ANTHROPIC_WORKSPACE_ID` was added to `.env` for this.
- Consequence: **no Claude call has been made live.** Vision calibration, the low-confidence
  OCR fallback and caption writing are implemented and tested against mocked responses
  only. Everything else was verified on the local route.
- When Claude is used, requests go to `CLAUDE_MODEL` with structured outputs
  (`output_config.format`), adaptive thinking at low/medium effort, and server-side refusal
  fallback (`fallbacks: "default"`), falling back to a plain request if the account does
  not accept that beta.
