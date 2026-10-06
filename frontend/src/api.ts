import type {
  BrowseResult,
  Calibration,
  Clip,
  ExportRecord,
  FieldRead,
  Game,
  Health,
  Job,
  JobOptions,
  RunStartPreview,
  Sport,
  StartSpec,
  TemplateRecord,
} from "./types";

/**
 * Where the engine (API + worker) lives.
 *
 * Run locally, the page and the API share an origin and this stays "". A hosted copy of
 * the frontend (e.g. on Vercel) has no API of its own: it talks to the engine running on
 * this computer, at http://127.0.0.1:8000 unless another address was saved.
 */
const STORAGE_KEY = "possession-cut.engine";
const DEFAULT_LOCAL = "http://127.0.0.1:8000";
let base = "";

export function engineBase(): string {
  return base;
}

export function savedEngine(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveEngine(url: string): void {
  try {
    if (url) localStorage.setItem(STORAGE_KEY, url);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* private mode */
  }
}

async function probe(candidate: string): Promise<Health | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    const res = await fetch(`${candidate}/api/health`, { signal: controller.signal });
    if (!res.ok || !(res.headers.get("content-type") ?? "").includes("json")) return null;
    const data = (await res.json()) as Health;
    return data.ok ? data : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Find a reachable engine. Returns its health, or null if none answered. */
export async function connectEngine(preferred?: string): Promise<Health | null> {
  const envBase = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
  // Same origin first: that is the normal local setup (Vite proxy, or the API serving the
  // built app). Only a hosted copy of the page falls through to an explicit address.
  const candidates = [preferred, "", envBase, savedEngine(), DEFAULT_LOCAL, "http://localhost:8000"]
    .filter((c): c is string => c !== undefined)
    .map((c) => c.replace(/\/+$/, ""));
  for (const candidate of [...new Set(candidates)]) {
    const health = await probe(candidate);
    if (health) {
      base = candidate;
      if (candidate) saveEngine(candidate);
      return health;
    }
  }
  return null;
}

/** Absolute URL for an API path or a media URL returned by the API. */
export function url(path: string): string {
  return `${base}${path}`;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url(path), {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json", ...init.headers } : init?.headers,
    });
  } catch {
    throw new ApiError(0, "Cannot reach the engine. Is it still running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    // the session ran out (or was never there): the app shows the sign-in screen
    if (res.status === 401 && path !== "/api/login") window.dispatchEvent(new Event(SIGNED_OUT));
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

/** Fired on `window` when the engine says the session is not signed in. */
export const SIGNED_OUT = "possession-cut:signed-out";

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
const qs = (params: Record<string, string | number | undefined | null>): string => {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") usp.set(k, String(v));
  const s = usp.toString();
  return s ? `?${s}` : "";
};

export interface UploadProgress {
  sent: number;
  total: number;
  bytesPerSecond: number;
}

interface UploadStatus {
  id: string;
  size: number;
  received: number;
  chunk_size: number;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Send a game file to the engine in pieces and get back the draft job made for it.
 *
 * A piece that fails is retried from wherever the engine says the file really ends, so
 * a blip in the connection costs seconds, not the whole upload.
 */
export async function uploadFile(
  file: File,
  onProgress: (p: UploadProgress) => void,
  signal?: AbortSignal,
): Promise<Job> {
  const started = await request<UploadStatus>("/api/uploads", json({ name: file.name, size: file.size }));
  const path = `/api/uploads/${started.id}`;
  const cancelled = () => new DOMException("Upload cancelled", "AbortError");
  const giveUp = async () => {
    try {
      await fetch(url(path), { method: "DELETE" });
    } catch {
      /* the engine drops unfinished uploads after a day anyway */
    }
  };

  let offset = started.received;
  let failures = 0;
  let speed = 0;
  let markTime = performance.now();
  let markSent = offset;
  while (offset < file.size) {
    if (signal?.aborted) {
      await giveUp();
      throw cancelled();
    }
    const piece = file.slice(offset, Math.min(file.size, offset + started.chunk_size));
    try {
      const res = await fetch(url(`${path}?offset=${offset}`), {
        method: "PUT",
        body: piece,
        headers: { "Content-Type": "application/octet-stream" },
        signal,
      });
      if (res.status === 409) {
        offset = (await res.json()).detail.received;
        continue;
      }
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail = typeof body.detail === "string" ? body.detail : detail;
        } catch {
          /* not json */
        }
        throw new ApiError(res.status, detail);
      }
      offset = (await res.json()).received;
      failures = 0;
    } catch (err) {
      if (signal?.aborted) {
        await giveUp();
        throw cancelled();
      }
      // the engine said no (wrong kind of file, upload gone): retrying will not help
      if (err instanceof ApiError && err.status >= 400 && err.status < 500) throw err;
      failures += 1;
      if (failures > 6) throw new ApiError(0, "The upload kept failing. Check that the engine is still running, then try again.");
      await sleep(Math.min(8000, 500 * 2 ** failures));
      try {
        offset = (await request<UploadStatus>(path)).received;
      } catch {
        /* still unreachable: the next attempt will find out */
      }
      continue;
    }
    const now = performance.now();
    const elapsed = (now - markTime) / 1000;
    if (elapsed >= 0.25) {
      const latest = (offset - markSent) / elapsed;
      speed = speed ? speed * 0.7 + latest * 0.3 : latest;
      markTime = now;
      markSent = offset;
    }
    onProgress({ sent: offset, total: file.size, bytesPerSecond: speed });
  }
  onProgress({ sent: file.size, total: file.size, bytesPerSecond: speed });
  return request<Job>(`${path}/complete`, { method: "POST" });
}

export interface JobSetup {
  source_path?: string;
  sport: string;
  game_id: string | null;
  game: Partial<Game>;
  team: string | null;
  start_spec: StartSpec;
  end_spec: StartSpec;
  options: JobOptions;
}

export const api = {
  health: () => request<Health>("/api/health"),
  login: (password: string) => request<{ authenticated: boolean }>("/api/login", json({ password })),
  logout: () => request<{ authenticated: boolean }>("/api/logout", { method: "POST" }),
  sports: () => request<Sport[]>("/api/sports"),
  browse: (path: string) => request<BrowseResult>(`/api/fs/browse${qs({ path })}`),
  inbox: () => request<{ dir: string; files: { name: string; path: string; size: number }[]; waiting: number; drafts: number[] }>("/api/inbox"),
  games: (sport: string, date: string, team?: string) => request<Game[]>(`/api/games${qs({ sport, date, team })}`),
  runStart: (sport: string, gameId: string, team: string, sourcePath?: string) =>
    request<RunStartPreview>(`/api/games/run-start${qs({ sport, game_id: gameId, team, source_path: sourcePath })}`),

  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: number) => request<Job>(`/api/jobs/${id}`),
  createJob: (setup: JobSetup & { source_path: string }) => request<Job>("/api/jobs", json(setup)),
  updateJob: (id: number, setup: Partial<JobSetup>) =>
    request<Job>(`/api/jobs/${id}`, { method: "PATCH", body: JSON.stringify(setup) }),
  deleteJob: (id: number) => request<{ deleted: number }>(`/api/jobs/${id}`, { method: "DELETE" }),
  cancelJob: (id: number) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),

  calibrate: (id: number, ignoreTemplates = false) =>
    request<Job>(`/api/jobs/${id}/calibrate`, json({ ignore_templates: ignoreTemplates })),
  calibration: (id: number) => request<Calibration>(`/api/jobs/${id}/calibration`),
  previewCalibration: (id: number, body: { bug: number[]; fields: Record<string, number[] | null>; frame: number }) =>
    request<{ visible: boolean; reads: Record<string, FieldRead>; crop: number[] }>(
      `/api/jobs/${id}/calibration/preview`,
      json(body),
    ),
  saveCalibration: (
    id: number,
    body: {
      bug: number[];
      fields: Record<string, number[] | null>;
      crop?: number[] | null;
      template_name?: string;
      broadcaster?: string;
      confirmed: boolean;
    },
  ) => request<Calibration>(`/api/jobs/${id}/calibration`, { method: "PUT", body: JSON.stringify(body) }),

  analyze: (id: number) => request<Job>(`/api/jobs/${id}/analyze`, { method: "POST" }),
  clips: (id: number) => request<Clip[]>(`/api/jobs/${id}/clips`),
  updateClip: (id: number, body: { enabled?: boolean; src_in?: number; src_out?: number }) =>
    request<Clip>(`/api/clips/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  resetClip: (id: number) => request<Clip>(`/api/clips/${id}/reset`, { method: "POST" }),
  nudgeClips: (jobId: number, body: { edge: "in" | "out"; delta: number; only_enabled?: boolean }) =>
    request<Clip[]>(`/api/jobs/${jobId}/clips/nudge`, { method: "POST", body: JSON.stringify(body) }),
  resetClips: (jobId: number) => request<Clip[]>(`/api/jobs/${jobId}/clips/reset`, { method: "POST" }),
  bulkClips: (jobId: number, updates: { id: number; enabled?: boolean; src_in?: number; src_out?: number }[]) =>
    request<Clip[]>(`/api/jobs/${jobId}/clips/bulk`, { method: "POST", body: JSON.stringify({ updates }) }),

  startExport: (id: number, body: { title: string; caption: string; audio_crossfade: boolean; options?: Record<string, unknown> }) =>
    request<ExportRecord>(`/api/jobs/${id}/export`, json(body)),
  exports: (id: number) => request<ExportRecord[]>(`/api/jobs/${id}/exports`),
  revealExport: (id: number) => request<{ revealed: string }>(`/api/exports/${id}/reveal`, { method: "POST" }),

  templates: () => request<TemplateRecord[]>("/api/templates"),
  deleteTemplate: (id: number) => request<{ deleted: number }>(`/api/templates/${id}`, { method: "DELETE" }),
};
