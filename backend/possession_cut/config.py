"""Runtime configuration, loaded from the environment and the repo-root .env."""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    # Needed only for keys that are not scoped to a workspace (sent as anthropic-workspace-id).
    anthropic_workspace_id: str = ""
    claude_model: str = "claude-sonnet-5-5"
    claude_budget_per_job_usd: float = 2.0
    ocr_sample_fps: float = 2.0

    allowed_roots: str = ""
    inbox_dir: str = "inbox"
    data_dir: str = "data"
    exports_dir: str = "exports"
    # Where files uploaded through the page are kept. Empty = data/uploads, or the
    # computer's local app-data folder when data/ sits inside a cloud-synced folder.
    uploads_dir: str = ""

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    analysis_workers: int = Field(default=0, ge=0)
    hwaccel: str = "none"
    # Extra browser origins allowed to call this API (comma separated), e.g. a hosted copy
    # of the frontend: https://your-app.vercel.app
    cors_origins: str = ""

    # -- resolved paths -------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def inbox_path(self) -> Path:
        return self._resolve(self.inbox_dir)

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def exports_path(self) -> Path:
        return self._resolve(self.exports_dir)

    @property
    def uploads_path(self) -> Path:
        if self.uploads_dir:
            return self._resolve(self.uploads_dir)
        if _is_synced(self.data_path):
            # a multi-gigabyte game file must not be pushed to OneDrive or Dropbox
            return _local_app_dir() / "uploads"
        return self.data_path / "uploads"

    @property
    def jobs_path(self) -> Path:
        return self.data_path / "jobs"

    @property
    def templates_path(self) -> Path:
        return self.data_path / "templates"

    @property
    def cache_path(self) -> Path:
        return self.data_path / "cache"

    @property
    def db_path(self) -> Path:
        return self.data_path / "possession_cut.db"

    @property
    def roots(self) -> list[Path]:
        """Folders the file picker may browse. Always includes inbox and exports."""
        raw = [r.strip() for r in self.allowed_roots.replace("\n", os.pathsep).split(os.pathsep)]
        roots = [Path(r).expanduser().resolve() for r in raw if r]
        if not roots:
            roots = [Path.home().resolve()]
        for extra in (self.inbox_path, self.exports_path, self.uploads_path):
            extra = extra.resolve()
            if not any(_is_within(extra, r) for r in roots):
                roots.append(extra)
        return roots

    @property
    def allowed_origins(self) -> list[str]:
        extra = [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]
        return ["http://localhost:5173", "http://127.0.0.1:5173", *extra]

    @property
    def heartbeat_path(self) -> Path:
        return self.data_path / "worker.heartbeat"

    @property
    def workers(self) -> int:
        if self.analysis_workers:
            return self.analysis_workers
        return max(1, min(6, (os.cpu_count() or 2) // 2))

    def ensure_dirs(self) -> None:
        for p in (
            self.inbox_path,
            self.data_path,
            self.jobs_path,
            self.templates_path,
            self.cache_path,
            self.exports_path,
            self.uploads_path,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: int | str) -> Path:
        p = self.jobs_path / str(job_id)
        p.mkdir(parents=True, exist_ok=True)
        return p


SYNCED_FOLDERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud drive", "mobile documents")


def _is_synced(path: Path) -> bool:
    """Does this path sit inside a folder a cloud client keeps in sync?"""
    return any(part.lower().startswith(SYNCED_FOLDERS) for part in path.resolve().parts)


def _local_app_dir() -> Path:
    """A per-user folder on this machine that no sync client touches."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "PossessionCut"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PossessionCut"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "possession-cut"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_allowed_path(path: Path, settings: Settings | None = None) -> bool:
    """True when ``path`` sits under one of the configured browse roots."""
    settings = settings or get_settings()
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    return any(_is_within(resolved, r) for r in settings.roots)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# -- ffmpeg discovery -----------------------------------------------------

_FALLBACK_BIN_DIRS = [
    # winget puts shims here; a shell started before the install will not have it on PATH yet
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
    Path("C:/ffmpeg/bin"),
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
]


def _find_binary(name: str, override: str) -> str:
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    exe = name + (".exe" if os.name == "nt" else "")
    for d in _FALLBACK_BIN_DIRS:
        candidate = d / exe
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"{name} was not found. Install ffmpeg 6+ (e.g. `winget install Gyan.FFmpeg` or "
        f"`brew install ffmpeg`) or set {name.upper()}_PATH in .env."
    )


@lru_cache
def ffmpeg_bin() -> str:
    return _find_binary("ffmpeg", get_settings().ffmpeg_path)


@lru_cache
def ffprobe_bin() -> str:
    return _find_binary("ffprobe", get_settings().ffprobe_path)
