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
