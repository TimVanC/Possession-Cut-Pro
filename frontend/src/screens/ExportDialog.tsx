import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, url } from "../api";
import { Modal, Note, ProgressBar, Spinner, Toggle } from "../components";
import { formatBytes, formatDate, formatDuration, useJobEvents } from "../lib";
import type { ExportRecord, Job } from "../types";

function Result({ record }: { record: ExportRecord }) {
  const [copied, setCopied] = useState(false);
  const reveal = useMutation({ mutationFn: () => api.revealExport(record.id) });
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(record.caption);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the text is selectable below */
    }
  };
  return (
    <div className="grid grid-cols-[240px_1fr] gap-5">
      <video src={url(record.url!)} controls playsInline className="w-full rounded-lg border border-ink-700 bg-black" style={{ aspectRatio: "9 / 16" }} />
      <div className="min-w-0 space-y-3">
        <div>
          <div className="text-base font-semibold">{record.title || "Export"}</div>
          <div className="num text-[13px] text-ink-400">
            {formatDuration(record.duration)} · {formatBytes(record.size_bytes)} · 1080×1920 · {formatDate(record.created_at)}
          </div>
        </div>
        <div className="num break-all rounded-lg bg-ink-850 px-3 py-2 text-xs text-ink-300">
          {record.path}
          <div className="mt-1 text-ink-400">with the cut list (.cutlist.json) and caption (.caption.txt) beside it</div>
        </div>
        <div>
          <span className="label">Suggested caption</span>
          <textarea readOnly className="field h-28 resize-none text-[13px] leading-relaxed" value={record.caption} />
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="btn" onClick={() => reveal.mutate()}>Reveal in folder</button>
          <button className="btn" onClick={copy}>{copied ? "Copied" : "Copy caption"}</button>
          <a className="btn" href={url(`${record.url}?download=true`)}>Download</a>
        </div>
        {reveal.error && <Note tone="error">{(reveal.error as Error).message}</Note>}
      </div>
    </div>
  );
}

export default function ExportDialog({
  job,
  enabledCount,
  runtime,
  onClose,
}: {
  job: Job;
  enabledCount: number;
  runtime: number;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const summary = job.summary;
  const defaultCaption = [job.game?.label, job.game?.date].filter(Boolean).join(" · ");
  const [title, setTitle] = useState(summary?.suggested_title ?? "");
  const [caption, setCaption] = useState(defaultCaption);
  const [crossfade, setCrossfade] = useState(true);
  const [showing, setShowing] = useState<number | null>(null);
  // sport-specific export toggles (baseball's home run trot)
  const sports = useQuery({ queryKey: ["sports"], queryFn: api.sports, staleTime: 60_000 });
  const sportOptions = (sports.data?.find((s) => s.key === job.sport)?.options ?? []).filter((o) => o.where === "export");
  const [extra, setExtra] = useState<Record<string, boolean>>({});

  const live = useJobEvents(job.id, true);
  const status = live?.status ?? job.status;
  const exporting = status === "exporting";
  const exportsQuery = useQuery({
    queryKey: ["exports", job.id],
    queryFn: () => api.exports(job.id),
    refetchInterval: exporting ? 1500 : false,
  });

  const start = useMutation({
    mutationFn: () =>
      api.startExport(job.id, {
        title,
        caption,
        audio_crossfade: crossfade,
        options: Object.fromEntries(sportOptions.map((o) => [o.key, extra[o.key] ?? o.default])),
      }),
    onSuccess: (record) => {
      setShowing(record.id);
      client.invalidateQueries({ queryKey: ["job", job.id] });
      client.invalidateQueries({ queryKey: ["exports", job.id] });
    },
  });

  // when the render finishes, fetch the finished record
  useEffect(() => {
    if (!exporting) {
      client.invalidateQueries({ queryKey: ["exports", job.id] });
      client.invalidateQueries({ queryKey: ["job", job.id] });
    }
  }, [exporting, client, job.id]);

  const records = exportsQuery.data ?? [];
  const current = records.find((r) => r.id === showing) ?? null;
  const past = records.filter((r) => r.status === "done" && r.id !== showing);

  return (
    <Modal title="Export" onClose={onClose} wide>
      {current && current.status === "done" && current.url ? (
        <div className="space-y-4">
          <Result record={current} />
          <button className="btn btn-sm" onClick={() => setShowing(null)}>Export again with different text</button>
        </div>
      ) : exporting || start.isPending || (current && current.status !== "failed") ? (
        <div className="py-10 text-center">
          <div className="text-base font-semibold">Rendering {formatDuration(runtime)} of video</div>
          <ProgressBar value={live?.progress ?? 0} className="mx-auto mt-6 max-w-md" />
          <p className="num mt-2 text-ink-300">{live?.message || "Starting"}</p>
          <p className="mt-6 text-xs text-ink-400">1080×1920, H.264 high profile, CRF 18, AAC 192 kbps. You can close this; the render keeps going.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {current?.status === "failed" && <Note tone="error">The render failed: {current.error}</Note>}
          {job.status === "failed" && job.error && !current && <Note tone="error">{job.error}</Note>}
          <div className="num text-ink-300">
            {enabledCount} clips · {formatDuration(runtime)} · 9:16
          </div>
          <div>
            <label className="label" htmlFor="title">Title (top bar)</label>
            <input id="title" className="field" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Knicks 29-point comeback vs Spurs" />
          </div>
          <div>
            <label className="label" htmlFor="caption">Small caption (bottom bar, optional)</label>
            <input id="caption" className="field" value={caption} onChange={(e) => setCaption(e.target.value)} placeholder="2026 NBA Finals, Game 4" />
          </div>
          <Toggle checked={crossfade} onChange={setCrossfade} label="Audio crossfade at cuts" hint="A short blend of the broadcast audio at each cut so there are no pops. Video cuts stay hard." />
          {sportOptions.map((o) => (
            <Toggle
              key={o.key}
              checked={extra[o.key] ?? o.default}
              onChange={(v) => setExtra((prev) => ({ ...prev, [o.key]: v }))}
              label={o.label}
              hint={o.hint}
            />
          ))}
          <p className="text-xs text-ink-400">Nothing is drawn over the video itself. Leave the title empty for plain black bars.</p>
          {start.error && <Note tone="error">{(start.error as Error).message}</Note>}
          <div className="flex justify-end gap-2">
            <button className="btn" onClick={onClose}>Close</button>
            <button className="btn btn-primary" disabled={enabledCount === 0 || start.isPending} onClick={() => start.mutate()}>
              {start.isPending ? <Spinner /> : "Render"}
            </button>
          </div>
          {past.length > 0 && (
            <div className="border-t border-ink-800 pt-4">
              <span className="label">Earlier exports</span>
              <ul className="space-y-1">
                {past.map((r) => (
                  <li key={r.id}>
                    <button className="flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left hover:bg-ink-800" onClick={() => setShowing(r.id)}>
                      <span className="flex-1 truncate">{r.title || r.file_name}</span>
                      <span className="num text-xs text-ink-400">{formatDuration(r.duration)} · {formatDate(r.created_at)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
