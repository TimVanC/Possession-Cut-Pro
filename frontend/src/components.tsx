import { useQuery } from "@tanstack/react-query";
import { type ReactNode, useEffect, useState } from "react";
import { api } from "./api";
import { STATUS_TEXT, formatBytes, formatEta } from "./lib";
import type { JobStatus, TaskProgress as TaskProgressData } from "./types";

/** True when the engine runs on a server: no file picker, no "reveal in folder". */
export function useHosted(): boolean {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: 30_000 });
  return !!health.data?.hosted;
}

const STATUS_STYLE: Record<JobStatus, string> = {
  draft: "bg-ink-700 text-ink-300",
  calibrating: "bg-sky-950 text-sky-300",
  ready: "bg-indigo-950 text-indigo-300",
  analyzing: "bg-sky-950 text-sky-300",
  review: "bg-amber-950 text-amber-300",
  exporting: "bg-sky-950 text-sky-300",
  done: "bg-emerald-950 text-emerald-300",
  failed: "bg-red-950 text-red-300",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${STATUS_STYLE[status]}`}>
      {STATUS_TEXT[status]}
    </span>
  );
}

export function ProgressBar({ value, className = "" }: { value: number; className?: string }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      className={`h-1.5 w-full overflow-hidden rounded-full bg-ink-700 ${className}`}
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="h-full rounded-full bg-court transition-[width] duration-300" style={{ width: `${pct}%` }} />
    </div>
  );
}

/**
 * A running task: the bar, what it is doing, roughly how long is left, and the steps of
 * the task with the current one marked.
 */
export function TaskProgress({
  task,
  compact = false,
  fallback = "Working",
}: {
  task: Pick<TaskProgressData, "progress" | "message" | "stage" | "steps" | "eta_seconds">;
  compact?: boolean;
  fallback?: string;
}) {
  const eta = formatEta(task.eta_seconds);
  const steps = task.steps ?? [];
  const waiting = steps.length > 0 && steps.every((s) => s.state === "todo");
  return (
    <div data-testid="task-progress">
      <ProgressBar value={task.progress} />
      <div className="num mt-1.5 flex items-baseline justify-between gap-4 text-xs text-ink-400">
        <span className="min-w-0 truncate text-left">
          {Math.round(task.progress * 100)}% · {task.message || task.stage || fallback}
        </span>
        <span className="shrink-0 text-ink-300" data-testid="task-eta">
          {eta ?? (waiting ? "waiting to start" : "estimating time")}
        </span>
      </div>
      {!compact && steps.length > 0 && (
        <ol className="mt-5 space-y-2 text-left text-[13px]" aria-label="Steps">
          {steps.map((step, i) => (
            <li key={step.label} className="flex items-center gap-3" data-state={step.state}>
              <span
                aria-hidden
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
                  step.state === "done"
                    ? "bg-court text-ink-950"
                    : step.state === "active"
                      ? "border-2 border-court text-court"
                      : "border border-ink-600 text-ink-400"
                }`}
              >
                {step.state === "done" ? "✓" : i + 1}
              </span>
              <span className={step.state === "active" ? "font-medium text-ink-100" : step.state === "done" ? "text-ink-300" : "text-ink-400"}>
                {step.label}
              </span>
              {step.state === "active" && <Spinner className="h-3 w-3" />}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const tone =
    value >= 0.85 ? "text-emerald-300 bg-emerald-950" : value >= 0.65 ? "text-amber-300 bg-amber-950" : "text-red-300 bg-red-950";
  return (
    <span className={`num rounded px-1.5 py-0.5 text-[11px] font-semibold ${tone}`} title="Confidence in this clip">
      {pct}%
    </span>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-ink-600 border-t-court ${className}`}
      aria-label="Loading"
    />
  );
}

export function Note({ tone = "info", children }: { tone?: "info" | "warn" | "error"; children: ReactNode }) {
  const style =
    tone === "error"
      ? "border-red-900 bg-red-950/60 text-red-200"
      : tone === "warn"
        ? "border-amber-900 bg-amber-950/50 text-amber-200"
        : "border-ink-700 bg-ink-850 text-ink-300";
  return <div className={`rounded-lg border px-3 py-2 text-[13px] leading-relaxed ${style}`}>{children}</div>;
}

export function Modal({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6" onMouseDown={onClose}>
      <div
        className={`panel flex max-h-[90vh] w-full flex-col ${wide ? "max-w-5xl" : "max-w-2xl"} shadow-2xl`}
        onMouseDown={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={title}
      >
        <div className="flex items-center justify-between border-b border-ink-800 px-5 py-3">
          <h2 className="text-base font-semibold">{title}</h2>
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-5">{children}</div>
      </div>
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 py-1.5">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? "bg-court" : "bg-ink-600"}`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${checked ? "left-[18px]" : "left-0.5"}`}
        />
      </button>
      <span>
        <span className="block font-medium">{label}</span>
        {hint && <span className="block text-xs text-ink-400">{hint}</span>}
      </span>
    </label>
  );
}

/** Browse local folders through the API and pick a video file. Nothing is uploaded. */
export function FileBrowser({ onPick, onClose }: { onPick: (path: string) => void; onClose: () => void }) {
  const [path, setPath] = useState("");
  const { data, error, isLoading } = useQuery({ queryKey: ["browse", path], queryFn: () => api.browse(path) });
  const inbox = useQuery({ queryKey: ["inbox"], queryFn: api.inbox });
  return (
    <Modal title="Choose a game file" onClose={onClose}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <button className="btn btn-sm" onClick={() => setPath("")}>
          Roots
        </button>
        {inbox.data && (
          <button className="btn btn-sm" onClick={() => setPath(inbox.data.dir)}>
            Inbox
          </button>
        )}
        {data?.parent !== undefined && data.parent !== null && path && (
          <button className="btn btn-sm" onClick={() => setPath(data.parent ?? "")}>
            ↑ Up
          </button>
        )}
        <span className="num truncate text-ink-400" title={data?.path}>
          {data?.path || "Allowed folders"}
        </span>
      </div>
      {error && <Note tone="error">{(error as Error).message}</Note>}
      {isLoading && <Spinner />}
      <ul className="divide-y divide-ink-800 overflow-hidden rounded-lg border border-ink-800">
        {data?.entries.map((entry) => (
          <li key={entry.path}>
            <button
              className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-ink-800"
              onClick={() => (entry.is_dir ? setPath(entry.path) : onPick(entry.path))}
            >
              <span className="w-5 text-center text-ink-400">{entry.is_dir ? "▸" : "▶"}</span>
              <span className={`min-w-0 flex-1 truncate ${entry.is_dir ? "" : "font-medium text-ink-100"}`}>{entry.name}</span>
              {!entry.is_dir && <span className="num text-xs text-ink-400">{formatBytes(entry.size)}</span>}
            </button>
          </li>
        ))}
        {data && data.entries.length === 0 && (
          <li className="px-3 py-6 text-center text-ink-400">No folders or video files here.</li>
        )}
      </ul>
      <p className="mt-3 text-xs text-ink-400">
        Files are registered by path and read in place. Only folders under ALLOWED_ROOTS (your home folder by default) are
        listed.
      </p>
    </Modal>
  );
}
