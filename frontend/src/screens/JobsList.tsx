import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, url } from "../api";
import { Note, ProgressBar, Spinner, StatusBadge } from "../components";
import { UploadBox } from "../Upload";
import { formatBytes, formatDate, formatDuration, gameTitle, jobRoute } from "../lib";
import type { Job } from "../types";

function JobCard({ job }: { job: Job }) {
  const navigate = useNavigate();
  const client = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const remove = useMutation({
    mutationFn: () => api.deleteJob(job.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const cancel = useMutation({
    mutationFn: () => api.cancelJob(job.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const summary = job.summary;
  return (
    <li className="panel p-4 transition-colors hover:border-ink-700">
      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <button className="truncate text-left text-[15px] font-semibold hover:text-court" onClick={() => navigate(jobRoute(job))}>
              {job.source_name}
            </button>
            <StatusBadge status={job.status} />
            {job.from_inbox && <span className="rounded bg-ink-800 px-1.5 py-0.5 text-[11px] text-ink-300">inbox</span>}
            {job.uploaded && <span className="rounded bg-ink-800 px-1.5 py-0.5 text-[11px] text-ink-300">uploaded</span>}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[13px] text-ink-300">
            <span>{gameTitle(job)}</span>
            {job.team && <span>Following {job.team}</span>}
            {job.probe && (
              <span className="num text-ink-400">
                {formatDuration(job.probe.duration)} · {job.probe.display_width}×{job.probe.height} · {formatBytes(job.probe.size_bytes)}
              </span>
            )}
          </div>
          {summary && (job.status === "review" || job.status === "done" || job.status === "exporting") && (
            <div className="num mt-1 text-[13px] text-ink-400">
              {summary.clips} clips · {formatDuration(summary.runtime_seconds)} · from {summary.start_label.toLowerCase()}
            </div>
          )}
          {job.busy && (
            <div className="mt-3">
              <ProgressBar value={job.progress} />
              <div className="num mt-1 text-xs text-ink-400">
                {Math.round(job.progress * 100)}% · {job.message || job.stage}
              </div>
            </div>
          )}
          {job.status === "failed" && job.error && (
            <div className="mt-2">
              <Note tone="error">{job.error}</Note>
            </div>
          )}
          {!job.source_exists && (
            <div className="mt-2">
              <Note tone="warn">
                {job.uploaded ? "The uploaded game file is gone. Upload it again to keep working on this game." : `The source file is no longer at ${job.source_path}.`}
              </Note>
            </div>
          )}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <span className="text-xs text-ink-400">{formatDate(job.created_at)}</span>
          <div className="flex gap-2">
            {job.busy ? (
              <button className="btn btn-sm" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
                Cancel
              </button>
            ) : confirming ? (
              <>
                <button className="btn btn-sm btn-danger" onClick={() => remove.mutate()} disabled={remove.isPending}>
                  Delete job
                </button>
                <button className="btn btn-sm" onClick={() => setConfirming(false)}>
                  Keep
                </button>
              </>
            ) : (
              <button className="btn btn-sm btn-ghost text-ink-400" onClick={() => setConfirming(true)} title={job.uploaded ? "Deletes the job, its analysis and the uploaded game file. Exports stay." : "Deletes the job and its analysis. The game file and exports stay."}>
                Delete
              </button>
            )}
            <Link className="btn btn-sm btn-primary" to={jobRoute(job)}>
              Open
            </Link>
          </div>
          {remove.error && <span className="text-xs text-red-300">{(remove.error as Error).message}</span>}
        </div>
      </div>
    </li>
  );
}

function Templates() {
  const client = useQueryClient();
  const templates = useQuery({ queryKey: ["templates"], queryFn: api.templates });
  const remove = useMutation({
    mutationFn: (id: number) => api.deleteTemplate(id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["templates"] }),
  });
  if (!templates.data || templates.data.length === 0) return null;
  return (
    <section className="mt-10">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-400">Broadcaster templates</h2>
      <p className="mt-1 text-[13px] text-ink-400">
        Saved score bug layouts. A new job that shows one of these bugs skips detection.
      </p>
      <ul className="mt-3 grid gap-3 sm:grid-cols-2">
        {templates.data.map((t) => (
          <li key={t.id} className="panel flex items-center gap-3 p-3">
            {t.image && <img src={url(t.image)} alt="" className="h-9 max-w-[55%] rounded object-contain" />}
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">{t.name}</div>
              <div className="text-xs text-ink-400">
                {t.sport.toUpperCase()} · used {t.use_count}× · {t.source}
              </div>
            </div>
            <button className="btn btn-sm btn-ghost text-ink-400" onClick={() => remove.mutate(t.id)}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function JobsList() {
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: (query) => (query.state.data?.some((j) => j.busy) ? 1500 : 6000),
  });
  const inbox = useQuery({ queryKey: ["inbox"], queryFn: api.inbox, refetchInterval: 10000 });
  const navigate = useNavigate();
  const empty = jobs.data?.length === 0;
  return (
    <main>
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">{empty ? "Make your first cut" : "Jobs"}</h1>
        {inbox.data && inbox.data.waiting > 0 && (
          <span
            className="rounded-full bg-court/15 px-2.5 py-0.5 text-xs font-semibold text-court"
            title={`Files waiting in ${inbox.data.dir}`}
          >
            Inbox · {inbox.data.waiting}
          </span>
        )}
      </div>
      {empty && (
        <p className="mt-1 max-w-2xl text-ink-400">
          Upload a full game broadcast. The app reads the score bug, finds every scoring possession for your team, and
          cuts a vertical video with the dead time removed.
        </p>
      )}

      <div className="mt-5">
        <UploadBox compact={!empty} onBrowse={() => navigate("/new")} />
      </div>

      <div className="mt-6">
        {jobs.isLoading && <Spinner />}
        {jobs.error && <Note tone="error">{(jobs.error as Error).message}</Note>}
        <ul className="space-y-3">{jobs.data?.map((job) => <JobCard key={job.id} job={job} />)}</ul>
      </div>
      <Templates />
    </main>
  );
}
