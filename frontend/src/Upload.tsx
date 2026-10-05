import { useQueryClient } from "@tanstack/react-query";
import { type DragEvent, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { type UploadProgress, uploadFile } from "./api";
import { Note, ProgressBar } from "./components";
import { formatBytes } from "./lib";

const ACCEPT = "video/*,.mp4,.mkv,.ts,.m2ts,.mov,.m4v,.webm,.avi,.mpg,.mpeg";

function timeLeft(p: UploadProgress): string {
  if (p.bytesPerSecond <= 0) return "";
  const seconds = (p.total - p.sent) / p.bytesPerSecond;
  if (seconds < 5) return "a few seconds left";
  if (seconds < 90) return `${Math.round(seconds)} s left`;
  return `${Math.round(seconds / 60)} min left`;
}

/**
 * The front door: drop a game file or click to choose one. The file goes to the engine in
 * pieces, and when the last one lands the app opens the game setup for it.
 */
export function UploadBox({ compact = false, onBrowse }: { compact?: boolean; onBrowse?: () => void }) {
  const navigate = useNavigate();
  const client = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);
  const [over, setOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sending = file !== null;

  // a file dropped beside the box must not make the browser open it and leave the app
  useEffect(() => {
    const stop = (e: Event) => e.preventDefault();
    window.addEventListener("dragover", stop);
    window.addEventListener("drop", stop);
    return () => {
      window.removeEventListener("dragover", stop);
      window.removeEventListener("drop", stop);
    };
  }, []);

  useEffect(() => {
    if (!sending) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [sending]);

  useEffect(() => () => abort.current?.abort(), []);

  const start = async (picked: File) => {
    setError(null);
    setFile(picked);
    setProgress({ sent: 0, total: picked.size, bytesPerSecond: 0 });
    const controller = new AbortController();
    abort.current = controller;
    try {
      const job = await uploadFile(picked, setProgress, controller.signal);
      client.invalidateQueries({ queryKey: ["jobs"] });
      navigate(`/jobs/${job.id}/setup`);
    } catch (err) {
      if (!controller.signal.aborted) setError((err as Error).message || "The upload failed.");
    } finally {
      abort.current = null;
      setFile(null);
      setProgress(null);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && !sending) void start(dropped);
  };

  if (sending && progress) {
    const done = progress.sent >= progress.total;
    return (
      <div className="panel p-5" data-testid="upload-progress">
        <div className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="truncate font-medium">{file.name}</div>
            <div className="num mt-0.5 text-xs text-ink-400">
              {done ? (
                "Checking the file"
              ) : (
                <>
                  {formatBytes(progress.sent) || "0 B"} of {formatBytes(progress.total)}
                  {progress.bytesPerSecond > 0 && ` · ${formatBytes(progress.bytesPerSecond)}/s · ${timeLeft(progress)}`}
                </>
              )}
            </div>
          </div>
          {!done && (
            <button className="btn btn-sm" onClick={() => abort.current?.abort()}>
              Cancel
            </button>
          )}
        </div>
        <ProgressBar value={progress.total ? progress.sent / progress.total : 0} className="mt-3" />
        <p className="mt-2 text-xs text-ink-400">Keep this tab open until the upload finishes.</p>
      </div>
    );
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        data-testid="upload-box"
        aria-label="Upload a game video"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            input.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`cursor-pointer rounded-xl border-2 border-dashed text-center transition-colors ${
          compact ? "px-5 py-6" : "px-6 py-12"
        } ${over ? "border-court bg-court/10" : "border-ink-700 bg-ink-900/40 hover:border-ink-600 hover:bg-ink-900/70"}`}
      >
        <p className={compact ? "text-base font-medium" : "text-lg font-semibold"}>
          {over ? "Drop it to upload" : "Upload a game video"}
        </p>
        <p className="mx-auto mt-1 max-w-md text-[13px] text-ink-400">
          Drop the full broadcast here, or click to choose it. MP4, MKV or TS, 480p or better.
        </p>
        <span className="btn btn-primary mt-4 inline-flex">Choose a file</span>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          className="hidden"
          data-testid="upload-input"
          onChange={(e) => {
            const picked = e.target.files?.[0];
            e.target.value = "";
            if (picked) void start(picked);
          }}
        />
      </div>
      {onBrowse && (
        <p className="mt-2 text-xs text-ink-400">
          Is the file already on the computer running Possession Cut?{" "}
          <button className="underline decoration-dotted hover:text-ink-100" onClick={onBrowse}>
            Pick it from disk
          </button>{" "}
          and skip the copy.
        </p>
      )}
      {error && (
        <div className="mt-3">
          <Note tone="error">{error}</Note>
        </div>
      )}
    </div>
  );
}
