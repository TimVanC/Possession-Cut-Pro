import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, url } from "../api";
import { ConfidenceBadge, Note, Spinner, TaskProgress } from "../components";
import { clamp, formatClock, formatDuration, formatSeconds, periodLabel, useJobEvents, useLatest } from "../lib";
import type { Box, Clip, Job, JobSummary } from "../types";
import ExportDialog from "./ExportDialog";

const NUDGE = 0.5;
const KIND_TEXT: Record<string, string> = { free_throws: "FT", field_goal: "", touchdown: "TD", goal: "Goal", run: "Run", block: "BLK", steal: "STL" };
const DEFENSE = new Set(["block", "steal"]);
const RATES = [1, 1.5, 2] as const;
const HISTORY_CAP = 100;

type Mode = "clip" | "sequence" | "free";
type Filter = "all" | "team" | "opp" | "scores" | "ft" | "defense" | "warnings" | "low" | "off" | "edited";
type Sort = "game" | "confidence";
/** One clip's editable state, as undo and redo restore it. */
type Edit = { id: number; enabled: boolean; src_in: number; src_out: number };

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

/** Why the confidence is what it is, in the owner's words. */
function reasons(clip: Clip, summary: JobSummary | null): string[] {
  const out: string[] = [];
  const reads = (clip.events ?? []).map((e) => e.confidence).filter((c): c is number => typeof c === "number");
  if (reads.length && Math.min(...reads) < 0.9) out.push(`the bug read ${Math.round(Math.min(...reads) * 100)}% clear at the make`);
  if (!clip.pbp_event_id && summary?.pbp.available && clip.points > 0) out.push("no play-by-play match for this score");
  out.push(...clip.warnings);
  return out;
}

function formatPlayhead(t: number): string {
  return `${formatDuration(Math.floor(t))}.${Math.floor((t % 1) * 10)}`;
}

function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private window or blocked storage: a convenience only
  }
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
  const why = reasons(clip, job.summary);
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
          {clip.points > 0 && <span className="num rounded bg-court/15 px-1.5 text-[11px] font-bold text-court">+{clip.points}</span>}
          {kind && <span className="rounded bg-ink-700 px-1.5 text-[11px] font-semibold text-ink-300">{kind}</span>}
          {clip.edited && <span className="text-[11px] text-sky-300" title="In or out point was moved">edited</span>}
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
      <ConfidenceBadge value={clip.confidence} title={why.length ? why.join("\n") : "Read cleanly and matched to the play-by-play"} />
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
    refetchInterval: (q) => {
      const d = q.state.data;
      if (d?.busy) return 1500;
      // a preview copy on its way: pick it up when it lands
      if (d?.preview?.wanted && !d.preview.ready && !d.preview.failed) return 10_000;
      return false;
    },
  });
  const live = useJobEvents(jobId, !!job.data?.busy);
  const analyzing = job.data?.status === "analyzing";
  const clipsQuery = useQuery({
    queryKey: ["clips", jobId, job.data?.status === "analyzing" ? "pending" : job.data?.summary?.clips],
    queryFn: () => api.clips(jobId),
    enabled: !!job.data && !analyzing,
  });
  const clips = useMemo(() => clipsQuery.data ?? [], [clipsQuery.data]);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>("clip");
  const [playing, setPlaying] = useState(false);
  const [now, setNow] = useState(0);
  const [exportOpen, setExportOpen] = useState(false);
  const [panel, setPanel] = useState<"unmatched" | "warnings" | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [sort, setSort] = useState<Sort>("game");
  const [rate, setRate] = useState<number>(() => {
    const stored = Number(readStored("possession-cut.review-rate"));
    return (RATES as readonly number[]).includes(stored) ? stored : 1;
  });
  const [historyLen, setHistoryLen] = useState<[number, number]>([0, 0]);
  const video = useRef<HTMLVideoElement>(null);
  const history = useRef<{ before: Edit[]; after: Edit[] }[]>([]);
  const redoStack = useRef<{ before: Edit[]; after: Edit[] }[]>([]);

  const summary = job.data?.summary ?? null;
  const follow = summary?.follow_side;

  // the list as shown: a filter, and game order or least confident first
  const visible = useMemo(() => {
    const keep = (c: Clip): boolean => {
      switch (filter) {
        case "team":
          return c.team === follow;
        case "opp":
          return c.team !== follow;
        case "scores":
          return c.points > 0 && c.kind !== "free_throws";
        case "ft":
          return c.kind === "free_throws";
        case "defense":
          return DEFENSE.has(c.kind);
        case "warnings":
          return c.warnings.length > 0;
        case "low":
          return c.confidence < 0.7;
        case "off":
          return !c.enabled;
        case "edited":
          return c.edited;
        default:
          return true;
      }
    };
    const list = clips.filter(keep);
    if (sort === "confidence") list.sort((a, b) => a.confidence - b.confidence || a.order - b.order);
    return list;
  }, [clips, filter, sort, follow]);

  const current: Clip | undefined = clips.find((c) => c.id === selectedId) ?? visible[0] ?? clips[0];
  const state = useLatest({ clips, visible, mode, current });

  // -- edits, with undo ---------------------------------------------------------------
  const latestClips = useCallback((): Clip[] => {
    for (const [, data] of client.getQueriesData<Clip[]>({ queryKey: ["clips", jobId] })) if (data?.length) return data;
    return [];
  }, [client, jobId]);
  const snapshot = useCallback(
    (ids: number[]): Edit[] => {
      const by = new Map(latestClips().map((c) => [c.id, c]));
      return ids.flatMap((id) => {
        const c = by.get(id);
        return c ? [{ id, enabled: c.enabled, src_in: c.src_in, src_out: c.src_out }] : [];
      });
    },
    [latestClips],
  );
  const setAllClips = useCallback(
    (updated: Clip[]) => client.setQueriesData<Clip[]>({ queryKey: ["clips", jobId] }, () => updated),
    [client, jobId],
  );
  const mergeClips = useCallback(
    (result: Clip | Clip[]): Clip[] => {
      const merged = Array.isArray(result) ? result : latestClips().map((c) => (c.id === (result as Clip).id ? (result as Clip) : c));
      setAllClips(merged);
      return merged;
    },
    [latestClips, setAllClips],
  );

  /** Any clip change: snapshot the affected clips, apply, remember the before and after for undo. */
  const edit = useMutation({
    mutationFn: async ({ ids, run }: { ids: number[]; run: () => Promise<Clip | Clip[]>; record?: boolean }) => {
      const before = snapshot(ids);
      const result = await run();
      return { before, list: mergeClips(result), ids };
    },
    onSuccess: ({ before, ids }, vars) => {
      if (vars.record === false) return;
      history.current.push({ before, after: snapshot(ids) });
      if (history.current.length > HISTORY_CAP) history.current.shift();
      redoStack.current = [];
      setHistoryLen([history.current.length, 0]);
    },
  });
  const cancel = useMutation({ mutationFn: () => api.cancelJob(jobId), onSuccess: () => client.invalidateQueries({ queryKey: ["job", jobId] }) });

  const seek = useCallback((t: number) => {
    const v = video.current;
    if (!v) return;
    v.currentTime = Math.max(0, t);
    setNow(t);
  }, []);

  const selectById = useCallback(
    (id: number, opts: { play?: boolean; keepMode?: boolean } = {}) => {
      const clip = state.current.clips.find((c) => c.id === id);
      if (!clip) return;
      setSelectedId(id);
      writeStored(`possession-cut.review.${jobId}`, String(id));
      if (!opts.keepMode) setMode("clip");
      seek(clip.segments[0][0]);
      if (opts.play) void video.current?.play();
    },
    [jobId, seek, state],
  );

  /** Previous or next clip as listed (so a filter or sort narrows J and K too). */
  const step = useCallback(
    (delta: number, opts: { play?: boolean } = {}) => {
      const { visible: list, current: clip } = state.current;
      if (list.length === 0) return;
      const at = clip ? list.findIndex((c) => c.id === clip.id) : -1;
      const next = list[clamp((at < 0 ? 0 : at) + delta, 0, list.length - 1)];
      if (next) selectById(next.id, { play: opts.play });
    },
    [selectById, state],
  );

  const showEdge = useCallback(
    (clip: Clip, edge: "in" | "out") => {
      setMode("clip");
      // show the edge that moved: the start from its new in point, the end from a second and a half before it
      seek(edge === "in" ? clip.src_in : Math.max(clip.segments[clip.segments.length - 1][0], clip.src_out - 1.5));
      void video.current?.play();
    },
    [seek],
  );

  const nudge = useCallback(
    (edge: "in" | "out", delta: number) => {
      const clip = state.current.current;
      if (!clip) return;
      const body = edge === "in" ? { src_in: clip.src_in + delta } : { src_out: clip.src_out + delta };
      edit.mutate(
        { ids: [clip.id], run: () => api.updateClip(clip.id, body) },
        { onSuccess: ({ list }) => showEdge(list.find((c) => c.id === clip.id) ?? clip, edge) },
      );
    },
    [edit, showEdge, state],
  );

  /** The in or out point of the selected clip set to where the playhead is. */
  const setEdgeHere = useCallback(
    (edge: "in" | "out") => {
      const clip = state.current.current;
      const v = video.current;
      if (!clip || !v) return;
      const t = v.currentTime;
      const body = edge === "in" ? { src_in: t } : { src_out: t };
      edit.mutate(
        { ids: [clip.id], run: () => api.updateClip(clip.id, body) },
        { onSuccess: ({ list }) => showEdge(list.find((c) => c.id === clip.id) ?? clip, edge) },
      );
    },
    [edit, showEdge, state],
  );

  // the same move for every clip in the cut; the player shows the selected clip's new edge
  const nudgeAll = useCallback(
    (edge: "in" | "out", delta: number) => {
      const ids = state.current.clips.filter((c) => c.enabled).map((c) => c.id);
      edit.mutate(
        { ids, run: () => api.nudgeClips(jobId, { edge, delta }) },
        {
          onSuccess: ({ list }) => {
            const clip = state.current.current;
            const mine = clip && list.find((c) => c.id === clip.id);
            if (mine) showEdge(mine, edge);
          },
        },
      );
    },
    [edit, jobId, showEdge, state],
  );

  const toggle = useCallback(
    (clip: Clip | undefined) => clip && edit.mutate({ ids: [clip.id], run: () => api.updateClip(clip.id, { enabled: !clip.enabled }) }),
    [edit],
  );
  const setShown = useCallback(
    (enabled: boolean) => {
      const list = state.current.visible;
      if (list.length === 0) return;
      edit.mutate({ ids: list.map((c) => c.id), run: () => api.bulkClips(jobId, list.map((c) => ({ id: c.id, enabled }))) });
    },
    [edit, jobId, state],
  );
  const resetOne = useCallback(
    (clip: Clip) => edit.mutate({ ids: [clip.id], run: () => api.resetClip(clip.id) }),
    [edit],
  );
  const resetAll = useCallback(() => {
    const ids = state.current.clips.filter((c) => c.edited).map((c) => c.id);
    edit.mutate({ ids, run: () => api.resetClips(jobId) });
  }, [edit, jobId, state]);

  const undo = useCallback(() => {
    const entry = history.current.pop();
    if (!entry) return;
    redoStack.current.push(entry);
    setHistoryLen([history.current.length, redoStack.current.length]);
    edit.mutate(
      { ids: entry.before.map((e) => e.id), run: () => api.bulkClips(jobId, entry.before), record: false },
      {
        onSuccess: () => {
          const clip = state.current.current;
          if (clip) seek(clip.segments[0][0]);
        },
      },
    );
  }, [edit, jobId, seek, state]);
  const redo = useCallback(() => {
    const entry = redoStack.current.pop();
    if (!entry) return;
    history.current.push(entry);
    setHistoryLen([history.current.length, redoStack.current.length]);
    edit.mutate({ ids: entry.after.map((e) => e.id), run: () => api.bulkClips(jobId, entry.after), record: false });
  }, [edit, jobId]);

  // -- playback -------------------------------------------------------------------------
  // keep playback inside the selected clip's kept segments; hop gaps; advance in sequence mode
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = video.current;
      const { current: clip, mode: m, clips: list } = state.current;
      if (v && clip && m !== "free") {
        const t = v.currentTime;
        const segs = clip.segments;
        const last = segs[segs.length - 1];
        if (t >= last[1] - 0.02) {
          // the sequence walks the whole cut in game order, whatever the list shows
          const at = list.findIndex((c) => c.id === clip.id);
          const next = m === "sequence" ? list.find((c, i) => i > at && c.enabled) : undefined;
          if (next) {
            setSelectedId(next.id);
            writeStored(`possession-cut.review.${jobId}`, String(next.id));
            v.currentTime = next.segments[0][0];
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
  }, [playing, state, jobId]);

  // first load: back on the clip left last time, else the first
  const ready = clips.length > 0 && !!job.data?.media_ready;
  useEffect(() => {
    if (!ready) return;
    const remembered = Number(readStored(`possession-cut.review.${jobId}`));
    const start = clips.find((c) => c.id === remembered) ?? clips[0];
    setSelectedId(start.id);
    seek(start.segments[0][0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // the preview copy landed: the player reloads with it, so put it back on the selected clip
  const mediaSource = job.data?.media_source ?? "source";
  useEffect(() => {
    const clip = state.current.current;
    if (ready && clip) seek(clip.segments[0][0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mediaSource]);

  // the chosen speed survives the player remounting for the preview copy
  useEffect(() => {
    if (video.current) video.current.playbackRate = rate;
    writeStored("possession-cut.review-rate", String(rate));
  }, [rate, mediaSource, ready]);

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

  /** Move the playhead by a little, paused, without the clip snapping it back. */
  const stepTime = useCallback(
    (delta: number) => {
      const v = video.current;
      if (!v) return;
      v.pause();
      setMode("free");
      const duration = Number.isFinite(v.duration) ? v.duration : Infinity;
      seek(clamp(v.currentTime + delta, 0, duration));
    },
    [seek],
  );

  const jumpToMake = useCallback(() => {
    const clip = state.current.current;
    const ev = clip?.events?.[0];
    if (!clip || !ev) return;
    setMode("clip");
    seek(Math.max(clip.segments[0][0], ev.t - 2));
    void video.current?.play();
  }, [seek, state]);

  const frame = 1 / (job.data?.probe?.fps || 30);

  // J/K previous/next, Space play/pause, X toggle, [ ] nudge in, { } nudge out, and the rest below
  useEffect(() => {
    if (exportOpen) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) && (target as HTMLInputElement).type !== "checkbox") return;
      // Alt + bracket moves every clip's in point; with Shift (a brace), every out point.
      // Matched on the physical key as well as the character: with Alt held, Mac keyboards
      // produce other characters and some senders leave the code blank.
      if (e.altKey && !e.metaKey && !e.ctrlKey) {
        const left = e.code === "BracketLeft" || e.key === "[" || e.key === "{";
        const right = e.code === "BracketRight" || e.key === "]" || e.key === "}";
        if (left || right) {
          nudgeAll(e.shiftKey || e.key === "{" || e.key === "}" ? "out" : "in", left ? -NUDGE : NUDGE);
          e.preventDefault();
          return;
        }
      }
      if ((e.ctrlKey || e.metaKey) && !e.altKey) {
        const k = e.key.toLowerCase();
        if (k === "z" && !e.shiftKey) undo();
        else if ((k === "z" && e.shiftKey) || k === "y") redo();
        else return;
        e.preventDefault();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const s = state.current;
      const wasPlaying = !!video.current && !video.current.paused;
      switch (e.key) {
        case "j":
        case "J":
          step(-1, { play: wasPlaying });
          break;
        case "k":
        case "K":
          step(1, { play: wasPlaying });
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
        case ",":
          stepTime(-frame);
          break;
        case ".":
          stepTime(frame);
          break;
        case "ArrowLeft":
          stepTime(e.shiftKey ? -5 : -1);
          break;
        case "ArrowRight":
          stepTime(e.shiftKey ? 5 : 1);
          break;
        case "i":
        case "I":
          setEdgeHere("in");
          break;
        case "o":
        case "O":
          setEdgeHere("out");
          break;
        case "m":
        case "M":
          jumpToMake();
          break;
        case "s":
        case "S":
          setRate((r) => RATES[(RATES.indexOf(r as (typeof RATES)[number]) + 1) % RATES.length]);
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exportOpen, frame, jumpToMake, nudge, nudgeAll, redo, setEdgeHere, state, step, stepTime, toggle, togglePlay, undo]);

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

  const sum = j.summary;
  const enabled = clips.filter((c) => c.enabled);
  const runtime = enabled.reduce((acc, c) => acc + c.duration, 0);
  const crop = j.calibration.crop as Box; // x, y, w, h
  const aspect = j.probe ? j.probe.display_width / j.probe.height : 16 / 9;
  const cropAspect = (crop[2] * aspect) / crop[3];
  const videoShare = Math.min(1, 9 / 16 / cropAspect);
  const warningCount = sum.warnings.length + clips.filter((c) => c.warnings.length > 0).length;
  const offset = current ? clipOffset(current, now) : 0;
  const bothTeams = clips.some((c) => c.team !== follow);
  const hasDefense = clips.some((c) => DEFENSE.has(c.kind));
  const chips: { key: Filter; label: string; show: boolean }[] = [
    { key: "all", label: "All", show: true },
    { key: "team", label: sum.team_abbr, show: bothTeams },
    { key: "opp", label: sum.opponent_abbr, show: bothTeams },
    { key: "scores", label: "Baskets", show: true },
    { key: "ft", label: "FT", show: clips.some((c) => c.kind === "free_throws") },
    { key: "defense", label: "BLK / STL", show: hasDefense },
    { key: "warnings", label: "⚠ Warnings", show: true },
    { key: "low", label: "Under 70%", show: true },
    { key: "off", label: "Off", show: true },
    { key: "edited", label: "Edited", show: true },
  ];
  const whyCurrent = current ? reasons(current, sum) : [];
  // where in the whole cut the sequence pass is
  const enabledBefore = current ? enabled.filter((c) => c.order < current.order) : [];
  const cutPosition = current && current.enabled ? enabledBefore.length + 1 : null;
  const cutElapsed = enabledBefore.reduce((acc, c) => acc + c.duration, 0) + Math.min(offset, current?.duration ?? 0);
  const [undoCount, redoCount] = historyLen;

  return (
    <main className="flex h-[calc(100vh-150px)] min-h-[560px] flex-col">
      {/* top bar */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pb-3">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold">{sum.suggested_title}</h1>
          <p className="truncate text-xs text-ink-400">
            {j.source_name} · from {sum.start_label.toLowerCase()} to {sum.end_label.toLowerCase()}
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
            className={`btn btn-sm ${sum.unmatched_pbp.length ? "border-amber-800 text-amber-300" : ""}`}
            onClick={() => setPanel(panel === "unmatched" ? null : "unmatched")}
            aria-expanded={panel === "unmatched"}
          >
            {sum.pbp.available ? `${sum.unmatched_pbp.length} unmatched plays` : "No play-by-play"}
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
          {!sum.pbp.available && <p className="text-ink-300">Play-by-play was not available for this job, so the cut comes from the score bug alone and clips are unlabeled.</p>}
          {sum.pbp.available && sum.unmatched_pbp.length === 0 && sum.unmatched_detected.length === 0 && (
            <p className="text-ink-300">Every score read off the bug matched a play in the play-by-play ({sum.pbp.source}), and every play in range was found on the bug.</p>
          )}
          {sum.unmatched_pbp.length > 0 && (
            <>
              <p className="mb-1 font-semibold">In the play-by-play but not found on the bug</p>
              <ul className="space-y-0.5">
                {sum.unmatched_pbp.map((p) => (
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
          {sum.unmatched_detected.length > 0 && (
            <>
              <p className="mb-1 mt-2 font-semibold">Read off the bug but not in the play-by-play</p>
              <ul className="space-y-0.5">
                {sum.unmatched_detected.map((p) => (
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
          {sum.warnings.map((w) => (
            <p key={w} className="text-amber-200">{w}</p>
          ))}
          {clips.filter((c) => c.warnings.length > 0).map((c) => (
            <button key={c.id} className="block w-full truncate text-left hover:text-white" onClick={() => selectById(c.id)}>
              <span className="num text-ink-400">Clip {c.order + 1}</span> · {c.warnings.join("; ")}
            </button>
          ))}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(300px,36%)] gap-5">
        {/* clip list */}
        <div className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="flex flex-wrap items-center gap-1.5 border-b border-ink-800 px-3 py-2 text-xs">
            {chips.filter((c) => c.show).map((c) => (
              <button
                key={c.key}
                className={`rounded-full border px-2.5 py-0.5 ${filter === c.key ? "border-court bg-court/15 text-court" : "border-ink-700 text-ink-300 hover:border-ink-500"}`}
                onClick={() => setFilter(c.key)}
                aria-pressed={filter === c.key}
              >
                {c.label}
              </button>
            ))}
            <button
              className={`ml-auto rounded-full border px-2.5 py-0.5 ${sort === "confidence" ? "border-court bg-court/15 text-court" : "border-ink-700 text-ink-300 hover:border-ink-500"}`}
              onClick={() => setSort(sort === "game" ? "confidence" : "game")}
              title="Sort the list so the clips that most need a look come first"
              aria-pressed={sort === "confidence"}
            >
              Least confident first
            </button>
          </div>
          {clipsQuery.isLoading && <div className="p-4"><Spinner /></div>}
          {clips.length === 0 && !clipsQuery.isLoading && (
            <div className="p-6 text-ink-300">
              No scoring possessions were found in the chosen range. Check the calibration (are the score boxes right?) or widen the start point in the game setup.
            </div>
          )}
          {clips.length > 0 && visible.length === 0 && <div className="p-6 text-ink-300">Nothing matches this filter.</div>}
          <ul className="min-h-0 flex-1 divide-y divide-ink-800/70 overflow-auto">
            {visible.map((clip) => (
              <ClipRow key={clip.id} clip={clip} job={j} selected={clip.id === current?.id} onSelect={() => selectById(clip.id, { play: true })} onToggle={() => toggle(clip)} />
            ))}
          </ul>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-ink-800 px-3 py-2 text-xs text-ink-400">
            <span className="num">{filter === "all" && sort === "game" ? `${clips.length} clips` : `showing ${visible.length} of ${clips.length}`}</span>
            {visible.length > 0 && (
              <>
                <button className="btn btn-sm" onClick={() => setShown(false)} disabled={edit.isPending || !visible.some((c) => c.enabled)}>
                  Turn these {visible.length} off
                </button>
                <button className="btn btn-sm" onClick={() => setShown(true)} disabled={edit.isPending || visible.every((c) => c.enabled)}>
                  Turn these {visible.length} on
                </button>
              </>
            )}
            <span className="ml-auto flex items-center gap-1">
              <button className="btn btn-sm" onClick={undo} disabled={undoCount === 0 || edit.isPending} title="Undo the last change to the clips (Ctrl+Z). History is kept until the page is reloaded.">
                Undo{undoCount ? ` (${undoCount})` : ""}
              </button>
              <button className="btn btn-sm" onClick={redo} disabled={redoCount === 0 || edit.isPending} title="Redo (Ctrl+Shift+Z)">
                Redo
              </button>
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-ink-800 px-3 py-1.5 text-[11px] text-ink-500">
            <span><kbd>J</kbd> <kbd>K</kbd> previous / next</span>
            <span><kbd>Space</kbd> play</span>
            <span><kbd>X</kbd> toggle</span>
            <span><kbd>[</kbd> <kbd>]</kbd> in point</span>
            <span><kbd>{"{"}</kbd> <kbd>{"}"}</kbd> out point</span>
            <span><kbd>Alt</kbd> + brackets: every clip</span>
            <span><kbd>I</kbd> <kbd>O</kbd> in / out at the playhead</span>
            <span><kbd>,</kbd> <kbd>.</kbd> a frame</span>
            <span><kbd>←</kbd> <kbd>→</kbd> a second (<kbd>Shift</kbd> five)</span>
            <span><kbd>M</kbd> the make</span>
            <span><kbd>S</kbd> speed</span>
            <span><kbd>Ctrl</kbd>+<kbd>Z</kbd> undo</span>
          </div>
        </div>

        {/* player */}
        <div className="flex min-h-0 flex-col items-center gap-3">
          <div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-ink-700 bg-black" style={{ aspectRatio: "9 / 16" }}>
            <div
              className="absolute inset-x-0 top-0 flex items-end justify-center px-3 pb-[4%] text-center font-bold leading-tight text-white"
              style={{ height: `${((1 - videoShare) / 2) * 100}%`, fontSize: "clamp(11px, 1.6vh, 18px)" }}
            >
              {sum.suggested_title}
            </div>
            <div className="absolute inset-x-0 overflow-hidden" style={{ top: `${((1 - videoShare) / 2) * 100}%`, height: `${videoShare * 100}%` }}>
              {j.media_ready ? (
                <video
                  key={j.media_source ?? "source"}
                  ref={video}
                  src={url(`/api/media/${jobId}/source?v=${j.media_source ?? "source"}`)}
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

          {j.preview?.wanted && !j.preview.ready && !j.preview.failed && (
            <p className="w-full max-w-md text-center text-xs text-ink-400">
              {j.preview.building
                ? `Preparing a smoother preview copy, ${Math.round((j.preview.progress ?? 0) * 100)}% done.`
                : "A smoother preview copy is queued."}
              {j.media_source === "source" && " Until then the original plays, and jumps between clips can take a moment."}
            </p>
          )}

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
                {(current.events ?? []).map((ev) => (
                  <button
                    key={ev.t}
                    className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink-950 bg-white"
                    style={{ left: `${clamp(clipOffset(current, ev.t) / Math.max(current.duration, 0.01), 0, 1) * 100}%` }}
                    title={`${ev.points > 0 ? `+${ev.points}` : "the play"} showed on the bug here${typeof ev.confidence === "number" ? ` (read ${Math.round(ev.confidence * 100)}% clear)` : ""}. Click to watch the make.`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setMode("clip");
                      seek(Math.max(current.segments[0][0], ev.t - 2));
                      void video.current?.play();
                    }}
                    aria-label="Watch the make"
                  />
                ))}
              </div>
              <div className="num flex items-center justify-between text-xs text-ink-400">
                <span>
                  clip {current.order + 1} · {formatSeconds(Math.min(offset, current.duration))} / {formatSeconds(current.duration)}
                  {current.segments.length > 1 && ` · ${current.segments.length} segments`}
                  {mode === "sequence" && cutPosition !== null && ` · ${cutPosition} of ${enabled.length} in the cut · ${formatDuration(cutElapsed)} of ${formatDuration(runtime)}`}
                </span>
                <span>{mode === "free" ? `free play ${formatPlayhead(now)}` : `source ${formatPlayhead(now)}`}</span>
              </div>
              {whyCurrent.length > 0 && (
                <p className="text-xs text-amber-200/90" title="Why the confidence is what it is">
                  {Math.round(current.confidence * 100)}%: {whyCurrent.join("; ")}
                </p>
              )}
              <div className="flex items-center justify-center gap-2">
                <div className="flex items-center gap-1" title="In point">
                  <span className="text-xs text-ink-400">In</span>
                  <button className="btn btn-sm num" onClick={() => nudge("in", -NUDGE)} aria-label="In point half a second earlier">−0.5</button>
                  <button className="btn btn-sm num" onClick={() => nudge("in", NUDGE)} aria-label="In point half a second later">+0.5</button>
                  <button className="btn btn-sm" onClick={() => setEdgeHere("in")} title="Start this clip where the playhead is (I)">here</button>
                </div>
                <button className="btn btn-primary w-20" onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
                <div className="flex items-center gap-1" title="Out point">
                  <button className="btn btn-sm" onClick={() => setEdgeHere("out")} title="End this clip where the playhead is (O)">here</button>
                  <button className="btn btn-sm num" onClick={() => nudge("out", -NUDGE)} aria-label="Out point half a second earlier">−0.5</button>
                  <button className="btn btn-sm num" onClick={() => nudge("out", NUDGE)} aria-label="Out point half a second later">+0.5</button>
                  <span className="text-xs text-ink-400">Out</span>
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-ink-400" title="Moves every clip that is in the cut by the same amount">
                <span>Every clip:</span>
                <div className="flex items-center gap-1">
                  <span>In</span>
                  <button className="btn btn-sm num" disabled={edit.isPending} onClick={() => nudgeAll("in", -NUDGE)} aria-label="Every in point half a second earlier">−0.5</button>
                  <button className="btn btn-sm num" disabled={edit.isPending} onClick={() => nudgeAll("in", NUDGE)} aria-label="Every in point half a second later">+0.5</button>
                </div>
                <div className="flex items-center gap-1">
                  <button className="btn btn-sm num" disabled={edit.isPending} onClick={() => nudgeAll("out", -NUDGE)} aria-label="Every out point half a second earlier">−0.5</button>
                  <button className="btn btn-sm num" disabled={edit.isPending} onClick={() => nudgeAll("out", NUDGE)} aria-label="Every out point half a second later">+0.5</button>
                  <span>Out</span>
                </div>
                {clips.some((c) => c.edited) && (
                  <button className="btn btn-sm" disabled={edit.isPending} onClick={resetAll} title="Every clip back to the in and out points the analysis found; which clips are on stays as it is">
                    Reset all edges
                  </button>
                )}
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
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
                <span className="flex items-center gap-0.5" title="Playback speed (S)">
                  {RATES.map((r) => (
                    <button key={r} className={`btn btn-sm num ${rate === r ? "border-court text-court" : ""}`} onClick={() => setRate(r)} aria-pressed={rate === r}>
                      {r}x
                    </button>
                  ))}
                </span>
                {current.edited && (
                  <button className="btn btn-sm" onClick={() => resetOne(current)}>Reset this clip</button>
                )}
              </div>
              {edit.error && <Note tone="error">{(edit.error as Error).message}</Note>}
            </div>
          )}
        </div>
      </div>

      {exportOpen && <ExportDialog job={j} enabledCount={enabled.length} runtime={runtime} onClose={() => setExportOpen(false)} />}
    </main>
  );
}
