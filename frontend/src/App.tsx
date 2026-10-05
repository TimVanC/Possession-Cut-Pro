import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api, connectEngine, engineBase, saveEngine, savedEngine } from "./api";
import { Note, Spinner } from "./components";
import CalibrationScreen from "./screens/Calibration";
import JobsList from "./screens/JobsList";
import NewJob from "./screens/NewJob";
import Review from "./screens/Review";
import type { Health } from "./types";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 2000 } },
});

/** Shown when no engine answers: the page is the screen, the engine on this computer does the work. */
function ConnectEngine({ onConnected }: { onConnected: (h: Health) => void }) {
  const [address, setAddress] = useState(savedEngine() || "http://127.0.0.1:8000");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const hosted = !["localhost", "127.0.0.1"].includes(window.location.hostname);

  // keep looking: the moment the engine is started, the page opens without a click
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const look = async () => {
      const health = await connectEngine();
      if (stopped) return;
      if (health) onConnected(health);
      else timer = setTimeout(look, 3000);
    };
    timer = setTimeout(look, 3000);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [onConnected]);

  const attempt = async () => {
    setBusy(true);
    setFailed(false);
    const health = await connectEngine(address.trim());
    setBusy(false);
    if (health) onConnected(health);
    else setFailed(true);
  };

  const code = "rounded bg-ink-800 px-1.5 py-0.5 text-ink-100";
  return (
    <main className="mx-auto flex min-h-full max-w-xl flex-col justify-center px-6 py-16">
      <h1 className="text-2xl font-semibold">
        Possession <span className="text-court">Cut</span>
      </h1>
      <p className="mt-4 text-lg font-medium">Start the engine to begin</p>
      <p className="mt-1 text-ink-300">
        This page is the screen. The video work is done by the Possession Cut engine on this computer, and it is not
        running yet.
      </p>
      <ol className="mt-5 list-decimal space-y-2 pl-5 text-ink-300">
        <li>
          Open the Possession Cut folder and double-click <code className={code}>start.cmd</code> (on a Mac, run{" "}
          <code className={code}>./dev.sh</code>).
        </li>
        <li>Leave it running. This page connects by itself and you can upload your game.</li>
      </ol>
      <p className="mt-4 text-[13px] text-ink-400">
        To skip this step for good, double-click <code className={code}>install-autostart.cmd</code> once. The engine
        then starts quietly every time you log in.
      </p>
      <div className="mt-6 flex items-center gap-3 text-[13px] text-ink-300" role="status">
        <Spinner /> Looking for the engine
      </div>

      <details className="mt-8 text-[13px] text-ink-400">
        <summary className="cursor-pointer hover:text-ink-100">Connection settings</summary>
        <label className="label mt-4" htmlFor="engine">
          Engine address
        </label>
        <div className="flex gap-2">
          <input id="engine" className="field" value={address} onChange={(e) => setAddress(e.target.value)} />
          <button className="btn btn-primary" onClick={attempt} disabled={busy}>
            {busy ? <Spinner /> : "Connect"}
          </button>
        </div>
        {hosted && (
          <p className="mt-3">
            This page is hosted at <b className="text-ink-300">{window.location.origin}</b>. The engine only answers
            sites listed under <code className={code}>CORS_ORIGINS</code> in its <code>.env</code> file, and Chrome asks
            once for permission to reach your local network.
          </p>
        )}
        {failed && (
          <div className="mt-3">
            <Note tone="error">
              No answer from {address}. Check that the engine is running{hosted ? " and that CORS_ORIGINS includes this site" : ""}.
            </Note>
          </div>
        )}
      </details>
    </main>
  );
}

function Banner({ health }: { health: Health }) {
  const live = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 8000, initialData: health });
  const h = live.data ?? health;
  const notes: { tone: "warn" | "error"; text: string }[] = [];
  if (!h.ffmpeg) notes.push({ tone: "error", text: "ffmpeg was not found. Install ffmpeg 6+ and restart." });
  if (!h.worker)
    notes.push({ tone: "warn", text: "The worker is not running, so queued jobs will wait. Start everything with start.cmd or ./dev.sh." });
  if (!h.claude.configured)
    notes.push({
      tone: "warn",
      text: h.claude.needs_workspace
        ? "Claude is off: the API key needs a workspace. Add ANTHROPIC_WORKSPACE_ID to .env (Claude Console, Settings, Workspaces) and restart. Until then calibration uses the on-device detector and captions use a template."
        : `Claude is off (${h.claude.note ?? "no API key"}). Calibration uses the on-device detector and captions use a template.`,
    });
  if (notes.length === 0) return null;
  return (
    <div className="space-y-2 px-6 pt-4">
      {notes.map((n) => (
        <Note key={n.text} tone={n.tone}>
          {n.text}
        </Note>
      ))}
    </div>
  );
}

function Shell({ health }: { health: Health }) {
  const location = useLocation();
  const wide = /\/(review|calibrate)$/.test(location.pathname);
  return (
    <div className="flex min-h-full flex-col">
      <header className="flex items-center gap-6 border-b border-ink-800 bg-ink-900/80 px-6 py-3">
        <Link to="/" className="text-lg font-semibold tracking-tight">
          Possession <span className="text-court">Cut</span>
        </Link>
        <nav className="flex gap-1 text-sm">
          <NavLink to="/" end className={({ isActive }) => `rounded-md px-3 py-1.5 ${isActive ? "bg-ink-800 text-white" : "text-ink-300 hover:text-white"}`}>
            Jobs
          </NavLink>
          <NavLink to="/new" className={({ isActive }) => `rounded-md px-3 py-1.5 ${isActive ? "bg-ink-800 text-white" : "text-ink-300 hover:text-white"}`}>
            New job
          </NavLink>
        </nav>
        <span className="ml-auto text-xs text-ink-400">
          v{health.version}
          {engineBase() && (
            <button
              className="ml-3 underline decoration-dotted hover:text-ink-100"
              title="Forget this engine address"
              onClick={() => {
                saveEngine("");
                window.location.reload();
              }}
            >
              engine: {engineBase()}
            </button>
          )}
        </span>
      </header>
      <Banner health={health} />
      <div className={`mx-auto w-full flex-1 px-6 py-6 ${wide ? "max-w-[1500px]" : "max-w-5xl"}`}>
        <Routes>
          <Route path="/" element={<JobsList />} />
          <Route path="/new" element={<NewJob />} />
          <Route path="/jobs/:id/setup" element={<NewJob />} />
          <Route path="/jobs/:id/calibrate" element={<CalibrationScreen />} />
          <Route path="/jobs/:id/review" element={<Review />} />
          <Route path="*" element={<JobsList />} />
        </Routes>
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [checked, setChecked] = useState(false);
  useEffect(() => {
    connectEngine().then((h) => {
      setHealth(h);
      setChecked(true);
    });
  }, []);
  if (!checked)
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  if (!health) return <ConnectEngine onConnected={setHealth} />;
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Shell health={health} />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
