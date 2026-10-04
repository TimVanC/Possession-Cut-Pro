export type JobStatus =
  | "draft"
  | "calibrating"
  | "ready"
  | "analyzing"
  | "review"
  | "exporting"
  | "done"
  | "failed";

export type Box = [number, number, number, number];

export interface StartSpec {
  mode: "start" | "end" | "game_time" | "auto_run";
  period?: number;
  clock?: number;
}

export interface Game {
  game_id: string;
  date: string;
  away: string;
  home: string;
  away_name: string;
  home_name: string;
  away_score: number | null;
  home_score: number | null;
  status: string;
  label: string;
}

export interface JobOptions {
  include_free_throws?: boolean;
  include_and_one_ft?: boolean;
  include_opponent?: boolean;
  [key: string]: unknown;
}

export interface UnmatchedPlay {
  event_id: string;
  period: number;
  clock: number | null;
  team: string;
  points: number;
  description: string;
  t: number;
}

export interface JobSummary {
  follow_side: "away" | "home";
  team: string;
  team_abbr: string;
  opponent: string;
  opponent_abbr: string;
  clips: number;
  runtime_seconds: number;
  events_detected: number;
  start_label: string;
  end_label: string;
  run_start: { deficit: number; period: number; clock: number | null; description: string } | null;
  pbp: { available: boolean; source: string; matched?: number; unmatched_detected?: number; unmatched_pbp?: number };
  unmatched_pbp: UnmatchedPlay[];
  unmatched_detected: { t: number; team: string; points: number; period: number | null; clock: number | null }[];
  final_score: { away: number | null; home: number | null };
  suggested_title: string;
  warnings: string[];
  claude_spent_usd: number;
}

export interface Job {
  id: number;
  status: JobStatus;
  stage: string;
  progress: number;
  message: string;
  busy: boolean;
  error: string | null;
  source_path: string;
  source_name: string;
  source_exists: boolean;
  sport: string;
  game_id: string | null;
  game: Partial<Game>;
  team: string | null;
  start_spec: StartSpec;
  end_spec: StartSpec;
  options: JobOptions;
  template_id: number | null;
  from_inbox: boolean;
  probe: {
    duration: number;
    width: number;
    height: number;
    display_width: number;
    fps: number;
    video_codec: string;
    audio_codec: string | null;
    size_bytes: number;
    browser_playable: boolean;
    container: string;
  } | null;
  calibration: {
    confidence: number;
    source: string;
    confirmed: boolean;
    template_name: string;
    teams: { away?: string; home?: string };
    broadcaster: string;
    warnings: string[];
    crop: Box;
    bug: Box;
  } | null;
  summary: JobSummary | null;
  media_ready: boolean;
  claude_spent_usd: number;
  created_at: string;
  updated_at: string;
}

export interface FieldRead {
  text: string;
  conf: number;
  value: number | string | null;
  ok: boolean;
}

export interface CalFrame {
  index: number;
  time: number;
  file: string;
  url: string;
  visible: boolean;
  similarity: number;
  reads: Record<string, FieldRead>;
}

export interface Calibration {
  bug: Box;
  fields: Record<string, Box>;
  crop: Box; // x, y, w, h normalized
  frame_width: number;
  frame_height: number;
  source: string;
  confidence: number;
  teams: { away?: string; home?: string };
  broadcaster: string;
  template_id: number | null;
  template_name: string;
  frames: CalFrame[];
  checks: Record<string, number>;
  warnings: string[];
  confirmed: boolean;
}

export interface Clip {
  id: number;
  job_id: number;
  order: number;
  enabled: boolean;
  src_in: number;
  src_out: number;
  segments: [number, number][];
  auto_in: number;
  auto_out: number;
  duration: number;
  team: "away" | "home";
  period: number | null;
  clock: number | null;
  score_before: number | null;
  score_after: number | null;
  score_away: number | null;
  score_home: number | null;
  points: number;
  kind: string;
  scorer: string;
  description: string;
  confidence: number;
  pbp_event_id: string | null;
  warnings: string[];
  thumbnail: string | null;
  edited: boolean;
}

export interface ExportRecord {
  id: number;
  job_id: number;
  status: "pending" | "rendering" | "done" | "failed";
  title: string;
  path: string;
  file_name: string;
  caption: string;
  cutlist_path: string;
  caption_path: string;
  duration: number;
  size_bytes: number;
  error: string | null;
  url: string | null;
  created_at: string;
}

export interface Sport {
  key: string;
  name: string;
  teams: { abbr: string; name: string }[];
  period_label: string;
  periods: number;
  period_seconds: number;
  has_clock: boolean;
  fields: { name: string; required: boolean; description: string }[];
  options: { key: string; label: string; default: boolean; where: string }[];
}

export interface Health {
  ok: boolean;
  version: string;
  ffmpeg: boolean;
  worker: boolean;
  claude: {
    configured: boolean;
    model: string;
    budget_per_job_usd: number;
    note: string | null;
    needs_workspace: boolean;
  };
  sample_fps: number;
  inbox_dir: string;
  exports_dir: string;
}

export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  modified: number | null;
  is_video: boolean;
}

export interface BrowseResult {
  path: string;
  parent: string | null;
  roots: string[];
  entries: BrowseEntry[];
}

export interface TemplateRecord {
  id: number;
  name: string;
  sport: string;
  broadcaster: string;
  source: string;
  use_count: number;
  image: string | null;
  created_at: string;
}

export interface RunStartPreview {
  available: boolean;
  trailed?: boolean;
  reason?: string;
  label?: string;
  after?: string;
  deficit?: number;
  period?: number;
  clock?: number;
}

export interface JobEvent {
  id: number;
  status: JobStatus;
  stage: string;
  progress: number;
  message: string;
  busy: boolean;
  error: string | null;
}
