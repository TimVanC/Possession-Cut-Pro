"""Locate a clean bold sans font on this machine (export titles, synthetic bug)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_ASSETS = Path(__file__).parent / "assets" / "fonts"

_BOLD_CANDIDATES = [
    # bundled (optional)
    _ASSETS / "Inter-Bold.ttf",
    # Windows
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf",
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf",
    # macOS
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    # Linux / Docker
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
]

_REGULAR_CANDIDATES = [
    _ASSETS / "Inter-Regular.ttf",
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf",
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
]


@lru_cache
def font_path(bold: bool = True) -> Path | None:
    for candidate in _BOLD_CANDIDATES if bold else _REGULAR_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=256)
def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = font_path(bold) or font_path(not bold)
    if path is not None:
        return ImageFont.truetype(str(path), size=size)
    # Pillow's embedded scalable default; not bold, but always present.
    return ImageFont.load_default(size=size)
