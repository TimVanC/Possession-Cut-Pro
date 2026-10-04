import { useEffect, useRef, useState } from "react";
import { url } from "./api";
import type { Job, JobEvent, JobStatus } from "./types";

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "-";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}

export function formatSeconds(seconds: number): string {
  return `${seconds.toFixed(1)}s`;
}

export function formatClock(clock: number | null | undefined): string {
  if (clock === null || clock === undefined) return "";
  if (clock >= 60) return `${Math.floor(clock / 60)}:${String(Math.floor(clock % 60)).padStart(2, "0")}`;
  return clock.toFixed(1);
}

/** "9:40" or "38.4" -> seconds. Null when it is not a clock. */
export function parseClock(text: string): number | null {
  const t = text.trim();
  const m = /^(\d{1,2}):(\d{2})$/.exec(t);
  if (m) {
    const s = Number(m[2]);
    return s < 60 ? Number(m[1]) * 60 + s : null;
  }
  if (/^\d{1,2}(\.\d)?$/.test(t)) return Number(t);
  return null;
}

export function periodLabel(period: number | null | undefined, label = "Q", regulation = 4): string {
  if (!period) return "";
  if (period <= regulation) return `${label}${period}`;
  return period - regulation === 1 ? "OT" : `${period - regulation}OT`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

export const STATUS_TEXT: Record<JobStatus, string> = {
  draft: "Draft",
  calibrating: "Calibrating",
  ready: "Ready to analyze",
  analyzing: "Analyzing",
  review: "Ready for review",
  exporting: "Exporting",
  done: "Exported",
  failed: "Failed",
};

/** Where "Open" should take you for a job in a given state. */
export function jobRoute(job: Pick<Job, "id" | "status" | "calibration">): string {
  switch (job.status) {
    case "draft":
      return `/jobs/${job.id}/setup`;
    case "calibrating":
    case "ready":
      return `/jobs/${job.id}/calibrate`;
    case "failed":
      return job.calibration ? `/jobs/${job.id}/calibrate` : `/jobs/${job.id}/setup`;
    default:
      return `/jobs/${job.id}/review`;
  }
}

export function gameTitle(job: Pick<Job, "game" | "sport">): string {
  const g = job.game;
  if (!g || !g.away || !g.home) return "No game selected";
  const score = g.away_score != null && g.home_score != null ? ` ${g.away_score}-${g.home_score}` : "";
  return `${g.away} @ ${g.home}${score}${g.date ? ` · ${g.date}` : ""}`;
}

/**
 * Live job state over server-sent events. The stream closes itself when the job goes
 * idle; EventSource reconnects on its own while the component is mounted and `active`.
 */
export function useJobEvents(jobId: number | null, active: boolean): JobEvent | null {
  const [event, setEvent] = useState<JobEvent | null>(null);
  useEffect(() => {
    if (!jobId || !active) return;
    const source = new EventSource(url(`/api/jobs/${jobId}/events`));
    source.onmessage = (msg) => {
      try {
        setEvent(JSON.parse(msg.data) as JobEvent);
      } catch {
        /* ignore */
      }
    };
    return () => source.close();
  }, [jobId, active]);
  return event;
}

export function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

/** Keep the latest callback in a ref so event listeners never go stale. */
export function useLatest<T>(value: T) {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
