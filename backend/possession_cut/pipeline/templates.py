"""Broadcaster templates: a saved bug layout plus a reference crop to recognize it by."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sqlmodel import Session, select

from ..config import get_settings
from ..db import Template, utcnow
from ..sports.base import SportAdapter
from .bugreader import VISIBLE_THRESHOLD, BugReader

log = logging.getLogger(__name__)

ASPECT_TOLERANCE = 0.03


def load_reference(template: Template) -> tuple[np.ndarray | None, np.ndarray | None]:
    ref = cv2.imread(template.reference_image) if template.reference_image else None
    mask = cv2.imread(template.mask_image, cv2.IMREAD_GRAYSCALE) if template.mask_image else None
    return ref, mask


def match_template(
    template: Template, frames: list[np.ndarray | None], adapter: SportAdapter
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Does this template's bug appear in these frames? Returns its (reference, mask) if so."""
    usable = [f for f in frames if f is not None]
    if not usable or template.sport != adapter.key or not template.bug:
        return None
    h, w = usable[0].shape[:2]
    if abs(w / h - template.frame_aspect) > ASPECT_TOLERANCE * template.frame_aspect:
        return None
    ref, mask = load_reference(template)
    if ref is None:
        return None
    reader = BugReader(
        tuple(template.bug), {k: tuple(v) for k, v in template.fields.items()}, adapter.bug_fields,
        w, h, None, ref, mask,
    )
    x, y, rw, rh = reader.roi
    hits = sum(1 for f in usable if reader.similarity(f[y : y + rh, x : x + rw]) >= VISIBLE_THRESHOLD)
    needed = max(2, int(round(0.2 * len(usable))))
    log.info("template %s: %d/%d frames match (need %d)", template.name, hits, len(usable), needed)
    return (ref, mask) if hits >= needed else None


def _unique_name(session: Session, base: str, ignore_id: int | None = None) -> str:
    existing = {t.name for t in session.exec(select(Template)).all() if t.id != ignore_id}
    if base not in existing:
        return base
    n = 2
    while f"{base} ({n})" in existing:
        n += 1
    return f"{base} ({n})"


def default_name(broadcaster: str, adapter: SportAdapter) -> str:
    return f"{(broadcaster or 'Broadcast').strip()} {adapter.name} {datetime.now().year}"


def _write_images(template: Template, ref: np.ndarray | None, mask: np.ndarray | None) -> None:
    folder: Path = get_settings().templates_path
    folder.mkdir(parents=True, exist_ok=True)
    if ref is not None:
        path = folder / f"template_{template.id}_ref.png"
        cv2.imwrite(str(path), ref)
        template.reference_image = str(path)
    if mask is not None:
        path = folder / f"template_{template.id}_mask.png"
        cv2.imwrite(str(path), mask)
        template.mask_image = str(path)


def save_template(
    session: Session,
    cal,
    adapter: SportAdapter,
    ref: np.ndarray | None,
    mask: np.ndarray | None,
    name: str | None = None,
) -> Template:
    """Create a template from a calibration, or update the one it came from."""
    template = session.get(Template, cal.template_id) if cal.template_id else None
    if template is None:
        template = Template(
            name=_unique_name(session, name or default_name(cal.broadcaster, adapter)),
            sport=adapter.key,
        )
        session.add(template)
    elif name and name != template.name:
        template.name = _unique_name(session, name, ignore_id=template.id)
    template.broadcaster = cal.broadcaster or template.broadcaster
    template.bug = list(cal.bug)
    template.fields = {k: list(v) for k, v in cal.fields.items()}
    template.crop = list(cal.crop)
    template.frame_aspect = cal.frame_width / cal.frame_height
    template.source = cal.source if cal.source != "template" else template.source
    template.last_used_at = utcnow()
    template.use_count = (template.use_count or 0) + 1
    session.flush()
    _write_images(template, ref, mask)
    session.add(template)
    session.flush()
    cal.template_id = template.id
    cal.template_name = template.name
    return template


def delete_template(session: Session, template: Template) -> None:
    for path in (template.reference_image, template.mask_image):
        if path:
            Path(path).unlink(missing_ok=True)
    session.delete(template)


def templates_for(session: Session, sport: str) -> list[Template]:
    """Most recently used first, so the likeliest match is tried first."""
    rows = session.exec(select(Template).where(Template.sport == sport)).all()
    return sorted(rows, key=lambda t: t.last_used_at or t.created_at, reverse=True)
