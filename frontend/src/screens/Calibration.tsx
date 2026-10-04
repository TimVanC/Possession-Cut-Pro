import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, url } from "../api";
import { Note, ProgressBar, Spinner } from "../components";
import { clamp, formatDuration, useDebounced, useJobEvents } from "../lib";
import type { Box, Calibration, FieldRead } from "../types";

const FIELD_COLORS: Record<string, string> = {
  away_label: "#94a3b8",
  away_score: "#38bdf8",
  home_label: "#94a3b8",
  home_score: "#f472b6",
  period: "#a78bfa",
  clock: "#facc15",
  shot_clock: "#fb923c",
};
const BUG_COLOR = "#22c55e";
const CROP_COLOR = "#f58220";
const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
type Handle = (typeof HANDLES)[number] | "move";

const fieldLabel = (name: string) => name.replace(/_/g, " ");

interface EditorBox {
  key: string;
  box: Box;
  color: string;
  label?: string;
  dashed?: boolean;
  lockY?: boolean;
}

/**
 * A frame (or a zoomed part of it) with draggable, resizable boxes on top.
 * `view` is the part of the image shown, in normalized image coordinates.
 */
function BoxEditor({
  image,
  aspect,
  view,
  boxes,
  selected,
  onSelect,
  onChange,
}: {
  image: string;
  aspect: number; // frame width / height
  view: Box;
  boxes: EditorBox[];
  selected: string | null;
  onSelect: (key: string | null) => void;
  onChange: (key: string, box: Box) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ key: string; handle: Handle; start: [number, number]; box: Box; lockY: boolean } | null>(null);
  const [vx0, vy0, vx1, vy1] = view;
  const vw = vx1 - vx0;
  const vh = vy1 - vy0;

  const toImage = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = ref.current!.getBoundingClientRect();
    return [vx0 + ((e.clientX - rect.left) / rect.width) * vw, vy0 + ((e.clientY - rect.top) / rect.height) * vh];
  };

  const begin = (e: ReactPointerEvent, item: EditorBox, handle: Handle) => {
    e.stopPropagation();
    e.preventDefault();
    onSelect(item.key);
    drag.current = { key: item.key, handle, start: toImage(e), box: item.box, lockY: !!item.lockY };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };

  const move = (e: ReactPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const [px, py] = toImage(e);
    const dx = px - d.start[0];
    const dy = d.lockY ? 0 : py - d.start[1];
    let [x0, y0, x1, y1] = d.box;
    const minW = 0.004;
    const minH = 0.006;
    if (d.handle === "move") {
      const w = x1 - x0;
      const h = y1 - y0;
      x0 = clamp(x0 + dx, 0, 1 - w);
      y0 = clamp(y0 + dy, 0, 1 - h);
      x1 = x0 + w;
      y1 = y0 + h;
    } else {
      if (d.handle.includes("w")) x0 = clamp(x0 + dx, 0, x1 - minW);
      if (d.handle.includes("e")) x1 = clamp(x1 + dx, x0 + minW, 1);
      if (d.handle.includes("n")) y0 = clamp(y0 + dy, 0, y1 - minH);
      if (d.handle.includes("s")) y1 = clamp(y1 + dy, y0 + minH, 1);
    }
    onChange(d.key, [x0, y0, x1, y1]);
  };

  const end = () => {
    drag.current = null;
  };

  const style = (b: Box) => ({
    left: `${((b[0] - vx0) / vw) * 100}%`,
    top: `${((b[1] - vy0) / vh) * 100}%`,
    width: `${((b[2] - b[0]) / vw) * 100}%`,
    height: `${((b[3] - b[1]) / vh) * 100}%`,
  });

  const handleStyle = (h: Handle): React.CSSProperties => {
    const pos: React.CSSProperties = { position: "absolute", width: 9, height: 9, marginLeft: -4.5, marginTop: -4.5 };
    pos.left = h.includes("w") ? "0%" : h.includes("e") ? "100%" : "50%";
    pos.top = h.includes("n") ? "0%" : h.includes("s") ? "100%" : "50%";
    pos.cursor = h === "n" || h === "s" ? "ns-resize" : h === "e" || h === "w" ? "ew-resize" : h === "nw" || h === "se" ? "nwse-resize" : "nesw-resize";
    return pos;
  };

  return (
    <div
      ref={ref}
      className="relative w-full select-none overflow-hidden rounded-lg bg-black"
      style={{ aspectRatio: `${(vw * aspect) / vh}` }}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
      onPointerDown={() => onSelect(null)}
    >
      <img
        src={image}
        alt="Frame from the game"
        draggable={false}
        className="pointer-events-none absolute max-w-none"
        style={{ width: `${100 / vw}%`, left: `${(-vx0 / vw) * 100}%`, top: `${(-vy0 / vh) * 100}%` }}
      />
      {boxes.map((item) => {
        const isSel = item.key === selected;
        return (
          <div
            key={item.key}
            className="absolute"
            style={{
              ...style(item.box),
              border: `${isSel ? 2 : 1.5}px ${item.dashed ? "dashed" : "solid"} ${item.color}`,
              background: isSel ? `${item.color}22` : "transparent",
              cursor: "move",
              zIndex: isSel ? 20 : item.key === "bug" || item.key === "crop" ? 5 : 10,
            }}
            onPointerDown={(e) => begin(e, item, "move")}
            title={item.label}
          >
            {item.label && (
              <span
                className="num pointer-events-none absolute -top-[18px] left-0 whitespace-nowrap rounded-sm px-1 text-[10px] font-semibold leading-4"
                style={{ background: item.color, color: "#0a0b0d" }}
              >
                {item.label}
              </span>
            )}
            {isSel &&
              HANDLES.filter((h) => !item.lockY || h === "e" || h === "w").map((h) => (
                <span
                  key={h}
                  style={{ ...handleStyle(h), background: "#fff", border: `1.5px solid ${item.color}`, borderRadius: 2 }}
                  onPointerDown={(e) => begin(e, item, h)}
                />
              ))}
          </div>
        );
      })}
    </div>
  );
}

/** What the export will look like: the crop, scaled to full width, centred on a 9:16 black canvas. */
function CropPreview({ image, crop, aspect, title }: { image: string; crop: Box; aspect: number; title: string }) {
  const [cx, cy, cw, ch] = crop;
  const cropAspect = (cw * aspect) / ch;
  const videoShare = Math.min(1, 9 / 16 / cropAspect);
  return (
    <div className="relative mx-auto w-[190px] overflow-hidden rounded-lg border border-ink-700 bg-black" style={{ aspectRatio: "9 / 16" }}>
      <div className="absolute inset-x-0 flex items-end justify-center px-2 text-center text-[9px] font-bold leading-tight text-white" style={{ top: 0, height: `${((1 - videoShare) / 2) * 100}%`, paddingBottom: 6 }}>
        {title}
      </div>
      <div className="absolute inset-x-0 overflow-hidden" style={{ top: `${((1 - videoShare) / 2) * 100}%`, height: `${videoShare * 100}%` }}>
        <img
          src={image}
          alt="Crop preview"
          className="absolute max-w-none"
          style={{ width: `${100 / cw}%`, left: `${(-cx / cw) * 100}%`, top: `${(-cy / ch) * 100}%` }}
        />
      </div>
    </div>
  );
}

/** PRD crop rule, mirrored here so the preview follows the box while it is being dragged. */
function cropByRule(bug: Box, aspect: number): Box {
  const bugW = bug[2] - bug[0];
  const centre = (bug[0] + bug[2]) / 2;
  let w = Math.min(1, bugW / 0.78);
  if (w * aspect < 1) {
    w = Math.min(1, 1.18 / aspect);
    let x = 0.5 - w / 2;
    if (bug[0] < x) x = bug[0];
    if (bug[2] > x + w) x = bug[2] - w;
    return [clamp(x, 0, 1 - w), 0, w, 1];
  }
  return [clamp(centre - w / 2, 0, 1 - w), 0, w, 1];
}

export default function CalibrationScreen() {
  const jobId = Number(useParams().id);
  const navigate = useNavigate();
  const client = useQueryClient();

  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    refetchInterval: (q) => (q.state.data?.busy ? 1200 : false),
  });
  const calibrating = job.data?.status === "calibrating";
  const live = useJobEvents(jobId, !!job.data?.busy);
  const sports = useQuery({ queryKey: ["sports"], queryFn: api.sports, staleTime: 60_000 });
  const cal = useQuery({
    queryKey: ["calibration", jobId, job.data?.updated_at],
    queryFn: () => api.calibration(jobId),
    enabled: !!job.data && !calibrating && !!job.data.calibration,
  });

  const [frame, setFrame] = useState(0);
  const [bug, setBug] = useState<Box | null>(null);
  const [fields, setFields] = useState<Record<string, Box>>({});
  const [cropOverride, setCropOverride] = useState<Box | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [templateName, setTemplateName] = useState("");
  const [broadcaster, setBroadcaster] = useState("");
  const [loadedFor, setLoadedFor] = useState<Calibration | null>(null);
  const [dirty, setDirty] = useState(false);

  // take the server's calibration as the starting point whenever a new one arrives
  useEffect(() => {
    const c = cal.data;
    if (!c || c === loadedFor) return;
    setLoadedFor(c);
    setBug(c.bug);
    setFields(c.fields);
    setCropOverride(null);
    setBroadcaster(c.broadcaster ?? "");
    setTemplateName(c.template_name ?? "");
    setDirty(false);
    const firstVisible = c.frames.findIndex((f) => f.visible);
    setFrame(firstVisible >= 0 ? firstVisible : 0);
  }, [cal.data, loadedFor]);

  const c = cal.data;
  const aspect = c ? c.frame_width / c.frame_height : 16 / 9;
  const frameInfo = c?.frames[frame];
  const frameUrl = frameInfo ? url(frameInfo.url) : "";
  const sportFields = sports.data?.find((s) => s.key === job.data?.sport)?.fields ?? [];

  // live read-out for the boxes as drawn
  const draftKey = useDebounced(JSON.stringify({ bug, fields, frame }), 300);
  const preview = useQuery({
    queryKey: ["cal-preview", jobId, draftKey],
    queryFn: () => api.previewCalibration(jobId, { bug: bug!, fields, frame }),
    enabled: !!c && !!bug && dirty,
    placeholderData: (prev) => prev,
  });
  const reads: Record<string, FieldRead> = (dirty ? preview.data?.reads : frameInfo?.reads) ?? frameInfo?.reads ?? {};

  const crop: Box = useMemo(() => {
    if (cropOverride) return cropOverride;
    if (bug && dirty) return cropByRule(bug, aspect);
    return c?.crop ?? [0, 0, 1, 1];
  }, [cropOverride, bug, dirty, c, aspect]);

  const zoomView: Box = useMemo(() => {
    if (!bug) return [0, 0, 1, 1];
    const h = bug[3] - bug[1];
    const w = bug[2] - bug[0];
    return [clamp(bug[0] - w * 0.04, 0, 1), clamp(bug[1] - h * 0.9, 0, 1), clamp(bug[2] + w * 0.04, 0, 1), clamp(bug[3] + h * 0.6, 0, 1)];
  }, [bug]);

  const onBoxChange = (key: string, box: Box) => {
    setDirty(true);
    if (key === "bug" && bug) {
      const dx = box[0] - bug[0];
      const dy = box[1] - bug[1];
      const moved = Math.abs(box[2] - box[0] - (bug[2] - bug[0])) < 1e-6 && Math.abs(box[3] - box[1] - (bug[3] - bug[1])) < 1e-6;
      if (moved) {
        // dragging the whole bug carries its fields along
        const next: Record<string, Box> = {};
        for (const [name, f] of Object.entries(fields)) next[name] = [f[0] + dx, f[1] + dy, f[2] + dx, f[3] + dy];
        setFields(next);
      }
      setBug(box);
    } else if (key === "crop") {
      setCropOverride([box[0], 0, box[2] - box[0], 1]);
    } else {
      setFields((prev) => ({ ...prev, [key]: box }));
    }
  };

  const addField = (name: string) => {
    if (!bug) return;
    const w = (bug[2] - bug[0]) * 0.1;
    const cx = (bug[0] + bug[2]) / 2;
    const pad = (bug[3] - bug[1]) * 0.12;
    setFields((prev) => ({ ...prev, [name]: [cx - w / 2, bug[1] + pad, cx + w / 2, bug[3] - pad] }));
    setSelected(name);
    setDirty(true);
  };
  const removeField = (name: string) => {
    setFields((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
    setDirty(true);
  };

  const redetect = useMutation({
    mutationFn: () => api.calibrate(jobId, true),
    onSuccess: () => client.invalidateQueries({ queryKey: ["job", jobId] }),
  });
  const confirm = useMutation({
    mutationFn: async () => {
      await api.saveCalibration(jobId, {
        bug: bug!,
        fields,
        crop: cropOverride,
        template_name: templateName || undefined,
        broadcaster,
        confirmed: true,
      });
      await api.analyze(jobId);
    },
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["job", jobId] });
      client.invalidateQueries({ queryKey: ["jobs"] });
      navigate(`/jobs/${jobId}/review`);
    },
  });

  if (job.isLoading) return <Spinner />;
  if (job.error) return <Note tone="error">{(job.error as Error).message}</Note>;
  const j = job.data!;

  if (calibrating || (j.busy && j.stage !== "calibrated" && !c)) {
    const p = live ?? j;
    return (
      <main className="mx-auto max-w-xl py-16 text-center">
        <h1 className="text-xl font-semibold">Finding the score bug</h1>
        <p className="mt-1 text-ink-400">{j.source_name}</p>
        <ProgressBar value={p.progress} className="mt-8" />
        <p className="num mt-2 text-ink-300">{p.message || "Working"}</p>
        <p className="mt-6 text-xs text-ink-400">
          Twelve frames are sampled across the file. A saved template is tried first, then the bug is located and each field is read to check it.
        </p>
      </main>
    );
  }
  if (j.status === "failed" && !j.calibration)
    return (
      <main className="mx-auto max-w-xl space-y-4 py-10">
        <Note tone="error">Calibration failed: {j.error}</Note>
        <button className="btn" onClick={() => redetect.mutate()}>Try again</button>
      </main>
    );
  if (!j.calibration)
    return (
      <main className="mx-auto max-w-xl space-y-4 py-10">
        <Note>This job has no calibration yet.</Note>
        <Link className="btn btn-primary" to={`/jobs/${jobId}/setup`}>Fill in the game setup</Link>
      </main>
    );
  if (!c || !bug) return <Spinner />;

  const missing = sportFields.filter((f) => f.required && !fields[f.name]);
  const readLine = ["away_label", "away_score", "home_label", "home_score", "period", "clock", "shot_clock"]
    .filter((n) => reads[n]?.text)
    .map((n) => reads[n].text)
    .join("  ·  ");

  const zoomBoxes: EditorBox[] = [
    { key: "bug", box: bug, color: BUG_COLOR },
    ...Object.entries(fields).map(([name, box]) => ({
      key: name,
      box,
      color: FIELD_COLORS[name] ?? "#e5e7eb",
      label: `${fieldLabel(name)}${reads[name]?.text ? `: ${reads[name].text}` : ""}`,
    })),
  ];
  const cropBox: Box = [crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3]];

  return (
    <main>
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <h1 className="text-xl font-semibold">Calibration</h1>
          <p className="text-ink-400">
            {j.source_name} · bug found by{" "}
            {c.source === "template" ? `template “${c.template_name}”` : c.source === "claude" ? "Claude vision" : c.source === "manual" ? "hand" : "the on-device detector"}{" "}
            · confidence <span className="num text-ink-100">{Math.round(c.confidence * 100)}%</span>
          </p>
        </div>
        <div className="ml-auto flex gap-2">
          <button className="btn" onClick={() => redetect.mutate()} disabled={redetect.isPending} title="Ignore saved templates and locate the bug from scratch">
            Detect again
          </button>
          <button className="btn btn-primary" onClick={() => confirm.mutate()} disabled={confirm.isPending || missing.length > 0}>
            {confirm.isPending ? <Spinner /> : "Looks right, analyze"}
          </button>
        </div>
      </div>

      <div className="mt-3 space-y-2">
        {c.warnings.map((w) => (
          <Note key={w} tone="warn">{w}</Note>
        ))}
        {missing.length > 0 && <Note tone="warn">Still needed: {missing.map((f) => fieldLabel(f.name)).join(", ")}. Add them from the list on the right.</Note>}
        {confirm.error && <Note tone="error">{(confirm.error as Error).message}</Note>}
        {!frameInfo?.visible && <Note>The bug does not appear to be on screen in this frame (a commercial or a replay). Pick another frame below.</Note>}
      </div>

      <div className="mt-4 grid grid-cols-[minmax(0,1fr)_300px] gap-5">
        <div className="space-y-4">
          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-ink-400">
              <span>Score bug, zoomed. Drag a box or its handles to adjust.</span>
              <span className="num text-ink-100">{readLine}</span>
            </div>
            <BoxEditor image={frameUrl} aspect={aspect} view={zoomView} boxes={zoomBoxes} selected={selected} onSelect={setSelected} onChange={onBoxChange} />
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-400">
              Whole frame. <span style={{ color: BUG_COLOR }}>Green</span> is the bug, <span style={{ color: CROP_COLOR }}>orange dashes</span> are the export crop (drag its sides to override).
            </div>
            <BoxEditor
              image={frameUrl}
              aspect={aspect}
              view={[0, 0, 1, 1]}
              boxes={[
                { key: "crop", box: cropBox, color: CROP_COLOR, dashed: true, lockY: true },
                { key: "bug", box: bug, color: BUG_COLOR },
              ]}
              selected={selected}
              onSelect={setSelected}
              onChange={onBoxChange}
            />
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-400">Other moments in the file</div>
            <div className="grid grid-cols-12 gap-1.5">
              {c.frames.map((f, i) => (
                <button
                  key={f.file}
                  onClick={() => setFrame(i)}
                  className={`relative overflow-hidden rounded border-2 ${i === frame ? "border-court" : "border-transparent opacity-70 hover:opacity-100"}`}
                  title={`${formatDuration(f.time)}${f.visible ? "" : " (bug not visible)"}`}
                >
                  <img src={url(f.url)} alt="" className="block w-full" />
                  <span className={`absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full ${f.visible ? "bg-emerald-400" : "bg-ink-600"}`} />
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="space-y-4">
          <div className="panel p-3">
            <span className="label">9:16 preview</span>
            <CropPreview image={frameUrl} crop={crop} aspect={aspect} title={j.summary?.suggested_title ?? "Title goes here"} />
            <p className="num mt-2 text-center text-xs text-ink-400">
              bug fills {Math.round(((bug[2] - bug[0]) / crop[2]) * 100)}% of the crop width
              {cropOverride && (
                <button className="ml-2 underline hover:text-ink-100" onClick={() => setCropOverride(null)}>
                  reset to rule
                </button>
              )}
            </p>
          </div>

          <div className="panel p-3">
            <span className="label">Fields {preview.isFetching && <Spinner className="ml-1 !h-3 !w-3" />}</span>
            <ul className="space-y-1">
              {sportFields.map((f) => {
                const has = !!fields[f.name];
                const read = reads[f.name];
                const isLabel = f.name.endsWith("_label");
                const ok = has && (isLabel || read?.ok || (!f.required && !read?.text));
                return (
                  <li key={f.name}>
                    <div
                      className={`flex items-center gap-2 rounded-md px-2 py-1 ${selected === f.name ? "bg-ink-800" : ""}`}
                      onClick={() => has && setSelected(f.name)}
                      role="button"
                    >
                      <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: FIELD_COLORS[f.name] ?? "#e5e7eb" }} />
                      <span className="flex-1 capitalize">{fieldLabel(f.name)}</span>
                      {has ? (
                        <>
                          <span className={`num font-semibold ${ok ? "text-ink-100" : "text-red-300"}`}>{read?.text || (f.required ? "unreadable" : "blank")}</span>
                          {!f.required && (
                            <button className="text-xs text-ink-400 hover:text-red-300" onClick={(e) => { e.stopPropagation(); removeField(f.name); }} title="This bug does not show this field">
                              ✕
                            </button>
                          )}
                        </>
                      ) : (
                        <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); addField(f.name); }}>
                          Add
                        </button>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
            <p className="mt-2 text-xs text-ink-400">
              Values are what OCR reads in this frame. Score boxes need room for three digits.
            </p>
          </div>

          <div className="panel space-y-3 p-3">
            <div>
              <label className="label" htmlFor="broadcaster">Broadcaster</label>
              <input id="broadcaster" className="field" value={broadcaster} onChange={(e) => setBroadcaster(e.target.value)} placeholder="ESPN, ABC, TNT…" />
            </div>
            <div>
              <label className="label" htmlFor="tname">Template name</label>
              <input id="tname" className="field" value={templateName} onChange={(e) => setTemplateName(e.target.value)} placeholder={`${broadcaster || "Broadcast"} ${j.sport.toUpperCase()} ${new Date().getFullYear()}`} />
            </div>
            <p className="text-xs text-ink-400">Saved when you confirm, and reused automatically the next time this bug shows up.</p>
          </div>

          <div className="num text-xs text-ink-400">
            bug on screen in {c.checks.frames_with_bug ?? 0} of {c.checks.frames_sampled ?? c.frames.length} sampled frames · clock read {Math.round((c.checks.clock_read_rate ?? 0) * 100)}% · scores read {Math.round((c.checks.score_read_rate ?? 0) * 100)}%
          </div>
        </aside>
      </div>
    </main>
  );
}
