import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, url } from "../api";
import { ConfidenceBadge, Note, Spinner, TaskProgress } from "../components";
import { clamp, formatClock, formatDuration, formatSeconds, periodLabel, useJobEvents, useLatest } from "../lib";
import type { Box, Clip, Job } from "../types";
import ExportDialog from "./ExportDialog";

const NUDGE = 0.5;
const KIND_TEXT: Record<string, string> = { free_throws: "FT", field_goal: "", touchdown: "TD", goal: "Goal", run: "Run" };

type Mode = "clip" | "sequence" | "free";

function teamAbbr(job: Job, clip: Clip): string {
  const s = job.summary;
  if (!s) return clip.team;
  return clip.team === s.follow_side ? s.team_abbr : s.opponent_abbr;
}

/** Position inside a clip counting only its kept segments (gaps cut out). */
function clipOffset(clip: Clip, t: number): number {
  let acc = 0;
  for (const [a, b] of clip.segments) {
    if (t < a) return acc;
    if (t <= b) return acc + (t - a);
    acc += b - a;
  }
  return acc;
}

function timeAtOffset(clip: Clip, offset: number): number {
  let left = offset;
  for (const [a, b] of clip.segments) {
    if (left <= b - a) return a + left;
    left -= b - a;
  }
  return clip.segments[clip.segments.length - 1][1];
}

function ClipRow({
  clip,
  job,
  selected,
  onSelect,
  onToggle,
}: {
  clip: Clip;
  job: Job;
  selected: boolean;
  onSelect: () => void;
  onToggle: () => void;
}) {
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  const abbr = teamAbbr(job, clip);
  const kind = KIND_TEXT[clip.kind] ?? clip.kind;
  return (
    <li
      ref={ref}
      onClick={onSelect}
      className={`flex cursor-pointer items-center gap-3 border-l-2 px-3 py-2 ${selected ? "border-court bg-ink-800" : "border-transparent hover:bg-ink-850"} ${clip.enabled ? "" : "opacity-45"}`}
      aria-selected={selected}
    >
      <span className="num w-5 text-right text-xs text-ink-400">{clip.order + 1}</span>
      {clip.thumbnail ? (
        <img src={url(clip.thumbnail)} alt="" className="h-11 w-[52px] shrink-0 rounded object-cover" loading="lazy" />
      ) : (
        <span className="h-11 w-[52px] shrink-0 rounded bg-ink-700" />
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="num text-[13px] text-ink-300">
            {periodLabel(clip.period)} {formatClock(clip.clock)}
          </span>
          <span className="num font-semibold">
            {abbr} {clip.score_before} → {clip.score_after}
          </span>
          <span className="num rounded bg-court/15 px-1.5 text-[11px] font-bold text-court">+{clip.points}</span>
          {kind && <span className="rounded bg-ink-700 px-1.5 text-[11px] font-semibold text-ink-300">{kind}</span>}
          {clip.edited && <span className="text-[11px] text-sky-300" title="In or out point was nudged">edited</span>}
        </div>
        <div className="truncate text-[13px] text-ink-400" title={clip.description}>
          {clip.description || (clip.scorer ? clip.scorer : "No play-by-play label")}
        </div>
      </div>
      {clip.warnings.length > 0 && (
        <span className="text-amber-400" title={clip.warnings.join("\n")} aria-label="Has warnings">
          ⚠
        </span>
      )}
      <span className="num w-11 text-right text-[13px] text-ink-300">{formatSeconds(clip.duration)}</span>
      <ConfidenceBadge value={clip.confidence} />
      <input
        type="checkbox"
        className="h-4 w-4 accent-court"
        checked={clip.enabled}
        onClick={(e) => e.stopPropagation()}
        onChange={onToggle}
        aria-label={clip.enabled ? "Remove clip from the cut" : "Add clip to the cut"}
      />
    </li>
  );
}

export default function Review() {
  const jobId = Number(useParams().id);
  const client = useQueryClient();
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    refetchInterval: (q) => (q.state.data?.busy ? 1500 : false),
  });
  const live = useJobEvents(jobId, !!job.data?.busy);
  const analyzing = job.data?.status === "analyzing";
  const clipsQuery = useQuery({
    queryKey: ["clips", jobId, job.data?.status === "analyzing" ? "pending" : job.data?.summary?.clips],
    queryFn: () => api.clips(jobId),
    enabled: !!job.data && !analyzing,
  });
  const clips = useMemo(() => clipsQuery.data ?? [], [clipsQuery.data]);

  const [selected, setSelected] = useState(0);
  const [mode, setMode] = useState<Mode>("clip");
  const [playing, setPlaying] = useState(false);
  const [now, setNow] = useState(0);
  const [exportOpen, setExportOpen] = useState(false);
  const [panel, setPanel] = useState<"unmatched" | "warnings" | null>(null);
  const video = useRef<HTMLVideoElement>(null);

  const current: Clip | undefined = clips[Math.min(selected, clips.length - 1)];
  const state = useLatest({ clips, selected, mode, current });

  const setClipCache = (updated: Clip) =>
    client.setQueriesData<Clip[]>({ queryKey: ["clips", jobId] }, (old) => old?.map((c) => (c.id === updated.id ? updated : c)));

  const patch = useMutation({
    mutationFn: ({ id, body }: { id: number; body: { enabled?: boolean; src_in?: number; src_out?: number } }) => api.updateClip(id, body),
    onSuccess: setClipCache,
  });
  const reset = useMutation({ mutationFn: (id: number) => api.resetClip(id), onSuccess: setClipCache });
  const cancel = useMutation({ mutationFn: () => api.cancelJob(jobId), onSuccess: () => client.invalidateQueries({ queryKey: ["job", jobId] }) });

  const seek = useCallback((t: number) => {
    const v = video.current;
    if (!v) return;
    v.currentTime = Math.max(0, t);
    setNow(t);
  }, []);

  const select = useCallback(
    (index: number, opts: { play?: boolean; keepMode?: boolean } = {}) => {
      const list = state.current.clips;
      if (list.length === 0) return;
      const i = clamp(index, 0, list.length - 1);
      setSelected(i);
      if (!opts.keepMode) setMode("clip");
      seek(list[i].segments[0][0]);
      if (opts.play) void video.current?.play();
    },
    [seek, state],
  );

  // keep playback inside the selected clip's kept segments; hop gaps; advance in sequence mode
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = video.current;
      const { current: clip, mode: m, clips: list, selected: sel } = state.current;
      if (v && clip && m !== "free") {
        const t = v.currentTime;
        const segs = clip.segments;
        const last = segs[segs.length - 1];
        if (t >= last[1] - 0.02) {
          const next = m === "sequence" ? list.findIndex((c, i) => i > sel && c.enabled) : -1;
          if (next >= 0) {
            setSelected(next);
            v.currentTime = list[next].segments[0][0];
          } else {
            v.pause();
            v.currentTime = last[1];
            if (m === "sequence") setMode("clip");
          }
        } else {
          const inside = segs.some(([a, b]) => t >= a - 0.05 && t < b);
          if (!inside) {
            const ahead = segs.find(([a]) => a > t);
            v.currentTime = ahead ? ahead[0] : segs[0][0];
          }
        }
      }
      if (v) setNow(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, state]);

  // first load: park on the first clip
  const ready = clips.length > 0 && !!job.data?.media_ready;
  useEffect(() => {
    if (ready) seek(clips[0].segments[0][0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  const togglePlay = useCallback(() => {
    const v = video.current;
    const clip = state.current.current;
    if (!v) return;
    if (!v.paused) {
      v.pause();
      return;
    }
    if (clip && state.current.mode !== "free") {
      const last = clip.segments[clip.segments.length - 1];
      if (v.currentTime >= last[1] - 0.05 || v.currentTime < clip.segments[0][0] - 0.05) v.currentTime = clip.segments[0][0];
    }
    void v.play();
  }, [state]);

  const nudge = useCallback(
    (edge: "in" | "out", delta: number) => {
      const clip = state.current.current;
      if (!clip) return;
      const body = edge === "in" ? { src_in: clip.src_in + delta } : { src_out: clip.src_out + delta };
      patch.mutate(
        { id: clip.id, body },
        {
          onSuccess: (updated) => {
            setMode("clip");
            // show the edge that moved: the start from its new in point, the end from a second and a half before it
            seek(edge === "in" ? updated.src_in : Math.max(updated.segments[updated.segments.length - 1][0], updated.src_out - 1.5));
            void video.current?.play();
          },
        },
      );
    },
    [patch, seek, state],
  );

  const toggle = useCallback(
    (clip: Clip | undefined) => clip && patch.mutate({ id: clip.id, body: { enabled: !clip.enabled } }),
    [patch],
  );

  // J/K previous/next, Space play/pause, X toggle, [ ] nudge in, { } nudge out
  useEffect(() => {
    if (exportOpen) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) && (target as HTMLInputElement).type !== "checkbox") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const s = state.current;
      const wasPlaying = !!video.current && !video.current.paused;
      switch (e.key) {
        case "j":
        case "J":
          select(s.selected - 1, { play: wasPlaying });
          break;
        case "k":
        case "K":
          select(s.selected + 1, { play: wasPlaying });
          break;
        case " ":
          togglePlay();
          break;
        case "x":
        case "X":
          toggle(s.current);
          break;
        case "[":
          nudge("in", -NUDGE);
          break;
        case "]":
          nudge("in", NUDGE);
          break;
        case "{":
          nudge("out", -NUDGE);
          break;
        case "}":
          nudge("out", NUDGE);
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exportOpen, nudge, select, state, toggle, togglePlay]);

  if (job.isLoading) return <Spinner />;
  if (job.error) return <Note tone="error">{(job.error as Error).message}</Note>;
  const j = job.data!;

  if (analyzing) {
    const p = live ?? j;
    return (
      <main className="mx-auto max-w-xl py-16 text-center">
        <h1 className="text-xl font-semibold">Analyzing</h1>
        <p className="mt-1 text-ink-400">{j.source_name}</p>
        <div className="mx-auto mt-8 max-w-md">
          <TaskProgress task={p} />
        </div>
        <p className="mx-auto mt-6 max-w-sm text-xs text-ink-400">
          A full game takes roughly 5 to 10 minutes. You can leave this page; the work carries on.
        </p>
        <button className="btn mt-6" onClick={() => cancel.mutate()} disabled={cancel.isPending}>Cancel</button>
      </main>
    );
  }
  if (j.status === "failed" && !j.summary)
    return (
      <main className="mx-auto max-w-xl space-y-4 py-10">
        <Note tone="error">Analysis failed: {j.error}</Note>
        <div className="flex gap-2">
          <Link className="btn" to={`/jobs/${jobId}/calibrate`}>Back to calibration</Link>
          <Link className="btn" to={`/jobs/${jobId}/setup`}>Game setup</Link>
        </div>
      </main>
    );
  if (!j.summary || !j.calibration)
    return (
      <main className="mx-auto max-w-xl space-y-4 py-10">
        <Note>This job has not been analyzed yet.</Note>
        <Link className="btn btn-primary" to={`/jobs/${jobId}/calibrate`}>Go to calibration</Link>
      </main>
    );

  const summary = j.summary;
  const enabled = clips.filter((c) => c.enabled);
  const runtime = enabled.reduce((acc, c) => acc + c.duration, 0);
  const crop = j.calibration.crop as Box; // x, y, w, h
  const aspect = j.probe ? j.probe.display_width / j.probe.height : 16 / 9;
  const cropAspect = (crop[2] * aspect) / crop[3];
  const videoShare = Math.min(1, 9 / 16 / cropAspect);
  const warningCount = summary.warnings.length + clips.filter((c) => c.warnings.length > 0).length;
  const offset = current ? clipOffset(current, now) : 0;

  return (
    <main className="flex h-[calc(100vh-150px)] min-h-[560px] flex-col">
      {/* top bar */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pb-3">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold">{summary.suggested_title}</h1>
          <p className="truncate text-xs text-ink-400">
            {j.source_name} · from {summary.start_label.toLowerCase()} to {summary.end_label.toLowerCase()}
          </p>
        </div>
        <div className="num ml-auto flex items-center gap-5 text-[13px]">
          <span>
            <span className="text-lg font-semibold text-ink-100">{formatDuration(runtime)}</span> <span className="text-ink-400">runtime</span>
          </span>
          <span>
            <span className="text-lg font-semibold text-ink-100">{enabled.length}</span>
            <span className="text-ink-400"> of {clips.length} clips</span>
          </span>
          <button
            className={`btn btn-sm ${summary.unmatched_pbp.length ? "border-amber-800 text-amber-300" : ""}`}
            onClick={() => setPanel(panel === "unmatched" ? null : "unmatched")}
            aria-expanded={panel === "unmatched"}
          >
            {summary.pbp.available ? `${summary.unmatched_pbp.length} unmatched plays` : "No play-by-play"}
          </button>
          <button
            className={`btn btn-sm ${warningCount ? "border-amber-800 text-amber-300" : ""}`}
            onClick={() => setPanel(panel === "warnings" ? null : "warnings")}
            aria-expanded={panel === "warnings"}
          >
            {warningCount} warnings
          </button>
          <Link className="btn btn-sm" to={`/jobs/${jobId}/setup`}>Game setup</Link>
          <button className="btn btn-primary" onClick={() => setExportOpen(true)} disabled={enabled.length === 0 || (j.busy && j.status !== "exporting")}>
            {j.status === "exporting" ? "Exporting…" : "Export"}
          </button>
        </div>
      </div>

      {panel === "unmatched" && (
        <div className="panel mb-3 max-h-56 overflow-auto p-3 text-[13px]">
          {!summary.pbp.available && <p className="text-ink-300">Play-by-play was not available for this job, so the cut comes from the score bug alone and clips are unlabeled.</p>}
          {summary.pbp.available && summary.unmatched_pbp.length === 0 && summary.unmatched_detected.length === 0 && (
            <p className="text-ink-300">Every score read off the bug matched a play in the play-by-play ({summary.pbp.source}), and every play in range was found on the bug.</p>
          )}
          {summary.unmatched_pbp.length > 0 && (
            <>
              <p className="mb-1 font-semibold">In the play-by-play but not found on the bug</p>
              <ul className="space-y-0.5">
                {summary.unmatched_pbp.map((p) => (
                  <li key={p.event_id} className="flex items-center gap-3">
                    <span className="num w-20 text-ink-400">{periodLabel(p.period)} {formatClock(p.clock)}</span>
                    <span className="flex-1 truncate">{p.team} +{p.points} · {p.description}</span>
                    <button
                      className="btn btn-sm"
                      onClick={() => {
                        setMode("free");
                        seek(Math.max(0, p.t - 8));
                        void video.current?.play();
                      }}
                    >
                      Jump to game time
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {summary.unmatched_detected.length > 0 && (
            <>
              <p className="mb-1 mt-2 font-semibold">Read off the bug but not in the play-by-play</p>
              <ul className="space-y-0.5">
                {summary.unmatched_detected.map((p) => (
                  <li key={p.t} className="num text-ink-300">
                    {periodLabel(p.period)} {formatClock(p.clock)} · {p.team} +{p.points} at {formatDuration(p.t)} in the file
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
      {panel === "warnings" && (
        <div className="panel mb-3 max-h-56 space-y-1.5 overflow-auto p-3 text-[13px]">
          {warningCount === 0 && <p className="text-ink-300">Nothing to flag.</p>}
          {summary.warnings.map((w) => (
            <p key={w} className="text-amber-200">{w}</p>
          ))}
          {clips.filter((c) => c.warnings.length > 0).map((c) => (
            <button key={c.id} className="block w-full truncate text-left hover:text-white" onClick={() => select(c.order)}>
              <span className="num text-ink-400">Clip {c.order + 1}</span> · {c.warnings.join("; ")}
            </button>
          ))}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(300px,36%)] gap-5">
        {/* clip list */}
        <div className="panel flex min-h-0 flex-col overflow-hidden">
          {clipsQuery.isLoading && <div className="p-4"><Spinner /></div>}
          {clips.length === 0 && !clipsQuery.isLoading && (
            <div className="p-6 text-ink-300">
              No scoring possessions were found in the chosen range. Check the calibration (are the score boxes right?) or widen the start point in the game setup.
            </div>
          )}
          <ul className="min-h-0 flex-1 divide-y divide-ink-800/70 overflow-auto">
            {clips.map((clip, i) => (
              <ClipRow key={clip.id} clip={clip} job={j} selected={i === selected} onSelect={() => select(i, { play: true })} onToggle={() => toggle(clip)} />
            ))}
          </ul>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-ink-800 px-3 py-2 text-xs text-ink-400">
            <span><kbd>J</kbd> <kbd>K</kbd> previous / next</span>
            <span><kbd>Space</kbd> play</span>
            <span><kbd>X</kbd> toggle</span>
            <span><kbd>[</kbd> <kbd>]</kbd> in point</span>
            <span><kbd>{"{"}</kbd> <kbd>{"}"}</kbd> out point</span>
          </div>
        </div>

        {/* player */}
        <div className="flex min-h-0 flex-col items-center gap-3">
          <div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-ink-700 bg-black" style={{ aspectRatio: "9 / 16" }}>
            <div
              className="absolute inset-x-0 top-0 flex items-end justify-center px-3 pb-[4%] text-center font-bold leading-tight text-white"
              style={{ height: `${((1 - videoShare) / 2) * 100}%`, fontSize: "clamp(11px, 1.6vh, 18px)" }}
            >
              {summary.suggested_title}
            </div>
            <div className="absolute inset-x-0 overflow-hidden" style={{ top: `${((1 - videoShare) / 2) * 100}%`, height: `${videoShare * 100}%` }}>
              {j.media_ready ? (
                <video
                  ref={video}
                  src={url(`/api/media/${jobId}/source`)}
                  preload="auto"
                  playsInline
                  className="absolute max-w-none"
                  style={{ width: `${100 / crop[2]}%`, left: `${(-crop[0] / crop[2]) * 100}%`, top: `${(-crop[1] / crop[3]) * 100}%` }}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onClick={togglePlay}
                />
              ) : (
                <div className="flex h-full items-center justify-center p-4 text-center text-xs text-ink-300">
                  This file cannot be played in the browser and its preview copy is missing. Re-run the analysis to make one. Export still works.
                </div>
              )}
            </div>
          </div>

          {current && (
            <div className="w-full max-w-md space-y-2">
              <div
                className="relative h-2 cursor-pointer rounded-full bg-ink-700"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  setMode("clip");
                  seek(timeAtOffset(current, ((e.clientX - rect.left) / rect.width) * current.duration));
                }}
                title="Position in this clip"
              >
                <div className="absolute inset-y-0 left-0 rounded-full bg-court" style={{ width: `${clamp(offset / Math.max(current.duration, 0.01), 0, 1) * 100}%` }} />
                {current.segments.slice(0, -1).map((seg, i) => {
                  const at = current.segments.slice(0, i + 1).reduce((acc, s) => acc + (s[1] - s[0]), 0);
                  return <span key={seg[1]} className="absolute inset-y-[-2px] w-0.5 bg-ink-100" style={{ left: `${(at / current.duration) * 100}%` }} title="A stretch was cut out here" />;
                })}
              </div>
              <div className="num flex items-center justify-between text-xs text-ink-400">
                <span>
                  clip {current.order + 1} · {formatSeconds(Math.min(offset, current.duration))} / {formatSeconds(current.duration)}
                  {current.segments.length > 1 && ` · ${current.segments.length} segments`}
                </span>
                <span>{mode === "free" ? "free play" : `source ${formatDuration(now)}`}</span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <div className="flex items-center gap-1" title="In point">
                  <span className="text-xs text-ink-400">In</span>
                  <button className="btn btn-sm num" onClick={() => nudge("in", -NUDGE)} aria-label="In point half a second earlier">−0.5</button>
                  <button className="btn btn-sm num" onClick={() => nudge("in", NUDGE)} aria-label="In point half a second later">+0.5</button>
                </div>
                <button className="btn btn-primary w-20" onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
                <div className="flex items-center gap-1" title="Out point">
                  <button className="btn btn-sm num" onClick={() => nudge("out", -NUDGE)} aria-label="Out point half a second earlier">−0.5</button>
                  <button className="btn btn-sm num" onClick={() => nudge("out", NUDGE)} aria-label="Out point half a second later">+0.5</button>
                  <span className="text-xs text-ink-400">Out</span>
                </div>
              </div>
              <div className="flex items-center justify-center gap-2">
                <button
                  className="btn btn-sm"
                  onClick={() => {
                    setMode("sequence");
                    seek(current.segments[0][0]);
                    void video.current?.play();
                  }}
                >
                  Play from here through the end
                </button>
                {current.edited && (
                  <button className="btn btn-sm" onClick={() => reset.mutate(current.id)}>Reset this clip</button>
                )}
              </div>
              {patch.error && <Note tone="error">{(patch.error as Error).message}</Note>}
            </div>
          )}
        </div>
      </div>

      {exportOpen && <ExportDialog job={j} enabledCount={enabled.length} runtime={runtime} onClose={() => setExportOpen(false)} />}
    </main>
  );
}
