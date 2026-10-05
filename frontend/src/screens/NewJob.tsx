import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, type JobSetup } from "../api";
import { FileBrowser, Note, Spinner, Toggle, useHosted } from "../components";
import { UploadBox } from "../Upload";
import { formatBytes, formatClock, formatDuration, parseClock, periodLabel } from "../lib";
import type { Game, StartSpec } from "../types";

type StartMode = "start" | "game_time" | "auto_run";
type EndMode = "end" | "game_time";

function dateFromName(name: string): string | null {
  const m = /(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])/.exec(name);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : null;
}

export default function NewJob() {
  const { id } = useParams();
  const jobId = id ? Number(id) : null;
  const navigate = useNavigate();
  const client = useQueryClient();

  const sports = useQuery({ queryKey: ["sports"], queryFn: api.sports, staleTime: 60_000 });
  const existing = useQuery({ queryKey: ["job", jobId], queryFn: () => api.job(jobId!), enabled: jobId !== null });

  const hosted = useHosted();
  const [browsing, setBrowsing] = useState(false);
  const [sourcePath, setSourcePath] = useState("");
  // no sport until one is chosen: the team list below belongs to a sport
  const [sport, setSport] = useState("");
  const [date, setDate] = useState("");
  const [teamFilter, setTeamFilter] = useState("");
  const [searched, setSearched] = useState<{ sport: string; date: string; team: string } | null>(null);
  const [game, setGame] = useState<Game | null>(null);
  const [team, setTeam] = useState("");
  const [startMode, setStartMode] = useState<StartMode>("auto_run");
  const [startPeriod, setStartPeriod] = useState(3);
  const [startClock, setStartClock] = useState("12:00");
  const [endMode, setEndMode] = useState<EndMode>("end");
  const [endPeriod, setEndPeriod] = useState(4);
  const [endClock, setEndClock] = useState("0:00");
  const [freeThrows, setFreeThrows] = useState(true);
  const [andOne, setAndOne] = useState(true);
  const [opponent, setOpponent] = useState(false);
  const [gameCamera, setGameCamera] = useState(true);
  const [loaded, setLoaded] = useState(false);

  // editing an existing job (an inbox draft, or changing options before re-analysis)
  useEffect(() => {
    const job = existing.data;
    if (!job || loaded) return;
    setLoaded(true);
    setSourcePath(job.source_path);
    // a file that was only just uploaded has had no sport chosen for it yet
    const untouched = job.status === "draft" && !job.game_id && !job.team && !job.calibration;
    setSport(untouched ? "" : job.sport);
    if (job.game?.game_id) {
      setGame(job.game as Game);
      setDate(job.game.date ?? "");
    }
    setTeam(job.team ?? "");
    const s = job.start_spec;
    setStartMode(s.mode === "game_time" || s.mode === "auto_run" ? s.mode : "start");
    if (s.mode === "game_time") {
      setStartPeriod(s.period ?? 1);
      setStartClock(formatClock(s.clock ?? 0));
    }
    const e = job.end_spec;
    setEndMode(e.mode === "game_time" ? "game_time" : "end");
    if (e.mode === "game_time") {
      setEndPeriod(e.period ?? 4);
      setEndClock(formatClock(e.clock ?? 0));
    }
    setFreeThrows(job.options.include_free_throws ?? true);
    setAndOne(job.options.include_and_one_ft ?? true);
    setOpponent(job.options.include_opponent ?? false);
    setGameCamera(job.options.trim_cutaways ?? true);
  }, [existing.data, loaded]);

  const sportInfo = sports.data?.find((s) => s.key === sport);
  const fileName = sourcePath.split(/[\\/]/).pop() ?? "";
  const nameDate = dateFromName(fileName);

  const games = useQuery({
    queryKey: ["games", searched],
    queryFn: () => api.games(searched!.sport, searched!.date, searched!.team || undefined),
    enabled: searched !== null,
    retry: 0,
  });

  const teamOptions = useMemo(() => {
    if (game)
      return [
        { value: game.away, label: `${game.away_name || game.away} (${game.away}, away)` },
        { value: game.home, label: `${game.home_name || game.home} (${game.home}, home)` },
      ];
    return [
      { value: "away", label: "First-listed team on the bug (away)" },
      { value: "home", label: "Second-listed team on the bug (home)" },
    ];
  }, [game]);

  useEffect(() => {
    if (!teamOptions.some((o) => o.value === team)) setTeam(teamOptions[teamOptions.length - 1].value);
  }, [teamOptions, team]);

  const runStart = useQuery({
    queryKey: ["run-start", sport, game?.game_id, team, sourcePath],
    queryFn: () => api.runStart(sport, game!.game_id, team, sourcePath || undefined),
    enabled: startMode === "auto_run" && game !== null && !!team,
    retry: 0,
  });

  const startClockValue = parseClock(startClock);
  const endClockValue = parseClock(endClock);
  const clockErrors =
    (startMode === "game_time" && startClockValue === null) || (endMode === "game_time" && endClockValue === null);

  const setup = (): JobSetup => {
    const start_spec: StartSpec =
      startMode === "game_time" ? { mode: "game_time", period: startPeriod, clock: startClockValue ?? 0 } : { mode: startMode };
    const end_spec: StartSpec =
      endMode === "game_time" ? { mode: "game_time", period: endPeriod, clock: endClockValue ?? 0 } : { mode: "end" };
    return {
      sport,
      game_id: game?.game_id ?? null,
      game: game ?? {},
      team,
      start_spec,
      end_spec,
      options: {
        include_free_throws: freeThrows,
        include_and_one_ft: andOne,
        include_opponent: opponent,
        trim_cutaways: gameCamera,
      },
    };
  };

  const calibrated = !!existing.data?.calibration;
  const submit = useMutation({
    mutationFn: async () => {
      if (jobId === null) {
        const job = await api.createJob({ ...setup(), source_path: sourcePath });
        return { id: job.id, to: "calibrate" };
      }
      await api.updateJob(jobId, setup());
      if (calibrated) {
        await api.analyze(jobId);
        return { id: jobId, to: "review" };
      }
      await api.calibrate(jobId);
      return { id: jobId, to: "calibrate" };
    },
    onSuccess: ({ id: newId, to }) => {
      client.invalidateQueries({ queryKey: ["jobs"] });
      client.invalidateQueries({ queryKey: ["job", newId] });
      navigate(`/jobs/${newId}/${to}`);
    },
  });

  const periods = Array.from({ length: (sportInfo?.periods ?? 4) + 2 }, (_, i) => i + 1);

  if (jobId !== null && existing.isLoading) return <Spinner />;
  if (existing.data?.busy)
    return <Note tone="warn">This job is busy ({existing.data.message || existing.data.stage}). Come back when it has finished, or cancel it from the jobs list.</Note>;

  return (
    <main className="mx-auto max-w-3xl">
      <h1 className="text-xl font-semibold">{jobId === null ? "New job" : "Game setup"}</h1>
      <p className="mt-1 text-ink-400">Upload the game, say which game and team, and choose where the cut starts.</p>

      {/* source */}
      <section className="panel mt-5 p-5">
        <span className="label">Game file</span>
        {sourcePath ? (
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">{fileName}</div>
              <div className="num truncate text-xs text-ink-400">{existing.data?.uploaded ? "Uploaded" : sourcePath}</div>
              {existing.data?.probe && (
                <div className="num mt-0.5 text-xs text-ink-400">
                  {formatDuration(existing.data.probe.duration)} · {existing.data.probe.display_width}×{existing.data.probe.height} ·{" "}
                  {formatBytes(existing.data.probe.size_bytes)}
                </div>
              )}
            </div>
            {jobId === null && (
              <button className="btn" onClick={() => setBrowsing(true)}>
                Change
              </button>
            )}
          </div>
        ) : (
          <UploadBox compact onBrowse={hosted ? undefined : () => setBrowsing(true)} />
        )}
      </section>

      {/* game */}
      <section className="panel mt-4 p-5">
        <div className="grid grid-cols-[150px_1fr_1fr_auto] items-end gap-3">
          <div>
            <label className="label" htmlFor="sport">Sport</label>
            <select
              id="sport"
              className="field"
              value={sport}
              onChange={(e) => {
                setSport(e.target.value);
                setTeamFilter("");
                setGame(null);
                setSearched(null);
              }}
            >
              <option value="">Choose a sport</option>
              {(sports.data ?? [{ key: "nba", name: "NBA" }]).map((s) => (
                <option key={s.key} value={s.key}>{s.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="date">Game date</label>
            <input id="date" type="date" className="field" value={date} onChange={(e) => setDate(e.target.value)} />
          </div>
          <div>
            <label className="label" htmlFor="teamfilter">Team (optional)</label>
            <select
              id="teamfilter"
              className="field"
              value={teamFilter}
              disabled={!sport}
              title={sport ? undefined : "Choose a sport to see its teams"}
              onChange={(e) => setTeamFilter(e.target.value)}
            >
              <option value="">{sport ? "Any team" : ""}</option>
              {sport &&
                sportInfo?.teams.map((t) => (
                  <option key={t.abbr} value={t.abbr}>{t.name}</option>
                ))}
            </select>
          </div>
          <button className="btn" disabled={!date || !sport} onClick={() => setSearched({ sport, date, team: teamFilter })}>
            Find games
          </button>
        </div>
        {nameDate && nameDate !== date && (
          <p className="mt-2 text-xs text-ink-400">
            The file name contains {nameDate}.{" "}
            <button className="underline hover:text-ink-100" onClick={() => setDate(nameDate)}>Use it</button> (file names are
            often wrong about the year, so check the result).
          </p>
        )}

        <div className="mt-4">
          {games.isFetching && <Spinner />}
          {games.error && <Note tone="error">{(games.error as Error).message}</Note>}
          {games.data && games.data.length === 0 && <Note>No games found on {searched?.date}{searched?.team ? ` for ${searched.team}` : ""}.</Note>}
          {games.data && games.data.length > 0 && (
            <ul className="divide-y divide-ink-800 overflow-hidden rounded-lg border border-ink-800">
              {games.data.map((g) => {
                const selected = game?.game_id === g.game_id;
                return (
                  <li key={g.game_id}>
                    <button
                      className={`flex w-full items-center gap-4 px-3 py-2.5 text-left ${selected ? "bg-court/10" : "hover:bg-ink-800"}`}
                      onClick={() => setGame(selected ? null : g)}
                      aria-pressed={selected}
                    >
                      <span className={`h-3.5 w-3.5 shrink-0 rounded-full border ${selected ? "border-court bg-court" : "border-ink-600"}`} />
                      <span className="num w-24 text-ink-400">{g.date}</span>
                      <span className="flex-1 font-medium">
                        {g.away_name || g.away} at {g.home_name || g.home}
                      </span>
                      <span className="num text-ink-300">
                        {g.away_score != null && g.home_score != null ? `${g.away} ${g.away_score}, ${g.home} ${g.home_score}` : g.status}
                      </span>
                      {g.label && <span className="max-w-[30%] truncate text-xs text-ink-400">{g.label}</span>}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          {game && !games.data && (
            <Note>
              Selected: {game.away_name || game.away} at {game.home_name || game.home}, {game.date}
              {game.label ? ` (${game.label})` : ""}.{" "}
              <button className="underline" onClick={() => setGame(null)}>Clear</button>
            </Note>
          )}
          {!game && (
            <p className="mt-2 text-xs text-ink-400">
              No game picked is fine: the cut is built from the score bug alone, and clips are not labeled with scorers.
            </p>
          )}
        </div>
      </section>

      {/* team, start, end */}
      <section className="panel mt-4 space-y-5 p-5">
        <div>
          <label className="label" htmlFor="team">Team to follow</label>
          <select id="team" className="field max-w-md" value={team} onChange={(e) => setTeam(e.target.value)}>
            {teamOptions.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        <div>
          <span className="label">Start point</span>
          <div className="space-y-2">
            {([
              ["auto_run", "Auto: start of the biggest run", "From the moment the team trailed by its largest deficit."],
              ["start", "Start of game", "Every scoring possession in the file."],
              ["game_time", "Game time", "A period and clock, for example Q3 2:26."],
            ] as const).map(([value, title, hint]) => (
              <label key={value} className="flex cursor-pointer items-start gap-3">
                <input type="radio" name="start" className="mt-1 accent-court" checked={startMode === value} onChange={() => setStartMode(value)} />
                <span>
                  <span className="font-medium">{title}</span>
                  <span className="block text-xs text-ink-400">{hint}</span>
                </span>
              </label>
            ))}
          </div>
          {startMode === "game_time" && (
            <div className="mt-3 flex items-center gap-2 pl-7">
              <select className="field w-28" value={startPeriod} onChange={(e) => setStartPeriod(Number(e.target.value))} aria-label="Start period">
                {periods.map((p) => (
                  <option key={p} value={p}>{periodLabel(p, sportInfo?.period_label, sportInfo?.periods)}</option>
                ))}
              </select>
              <input className={`field num w-28 ${startClockValue === null ? "border-red-700" : ""}`} value={startClock} onChange={(e) => setStartClock(e.target.value)} placeholder="2:26" aria-label="Start clock" />
              {startClockValue === null && <span className="text-xs text-red-300">Use M:SS, like 2:26</span>}
            </div>
          )}
          {startMode === "auto_run" && (
            <div className="mt-3 pl-7">
              {!game && <Note>Without a game the low point is read from the score bug during analysis.</Note>}
              {game && runStart.isFetching && <Spinner />}
              {game && runStart.data?.available && runStart.data.trailed && (
                <Note>
                  <b>{runStart.data.label}</b>
                  {runStart.data.after ? ` after “${runStart.data.after}”.` : "."} The cut starts with {team}&apos;s next possession.
                </Note>
              )}
              {game && runStart.data?.available && runStart.data.trailed === false && <Note>{runStart.data.label}</Note>}
              {game && runStart.data && !runStart.data.available && (
                <Note tone="warn">Play-by-play is not available right now ({runStart.data.reason}). The low point will be read from the score bug during analysis.</Note>
              )}
            </div>
          )}
        </div>

        <div>
          <span className="label">End point</span>
          <div className="flex flex-wrap items-center gap-5">
            <label className="flex cursor-pointer items-center gap-2">
              <input type="radio" name="end" className="accent-court" checked={endMode === "end"} onChange={() => setEndMode("end")} />
              End of game
            </label>
            <label className="flex cursor-pointer items-center gap-2">
              <input type="radio" name="end" className="accent-court" checked={endMode === "game_time"} onChange={() => setEndMode("game_time")} />
              Game time
            </label>
            {endMode === "game_time" && (
              <span className="flex items-center gap-2">
                <select className="field w-28" value={endPeriod} onChange={(e) => setEndPeriod(Number(e.target.value))} aria-label="End period">
                  {periods.map((p) => (
                    <option key={p} value={p}>{periodLabel(p, sportInfo?.period_label, sportInfo?.periods)}</option>
                  ))}
                </select>
                <input className={`field num w-28 ${endClockValue === null ? "border-red-700" : ""}`} value={endClock} onChange={(e) => setEndClock(e.target.value)} aria-label="End clock" />
              </span>
            )}
          </div>
        </div>

        <div>
          <span className="label">Include</span>
          <Toggle checked={freeThrows} onChange={setFreeThrows} label="Free throws" hint="Trimmed tight around each make." />
          <Toggle checked={andOne} onChange={setAndOne} label="And-one free throw" hint="Tacked onto its basket as a short trailing segment." />
          <Toggle checked={opponent} onChange={setOpponent} label="Opponent scores" hint="Both teams' scoring possessions, in game order." />
          <Toggle checked={gameCamera} onChange={setGameCamera} label="Game camera only" hint="Trims crowd shots and close-ups off the start and end of each clip." />
        </div>
      </section>

      {submit.error && (
        <div className="mt-4">
          <Note tone="error">{(submit.error as Error).message}</Note>
        </div>
      )}
      <div className="mt-5 flex items-center justify-end gap-3">
        <button className="btn" onClick={() => navigate("/")}>Cancel</button>
        {sourcePath && !sport && <span className="text-[13px] text-ink-400">Choose a sport to continue.</span>}
        <button className="btn btn-primary" disabled={!sourcePath || !sport || clockErrors || submit.isPending} onClick={() => submit.mutate()}>
          {submit.isPending ? <Spinner /> : jobId === null ? "Create job and find the score bug" : calibrated ? "Save and re-analyze" : "Save and find the score bug"}
        </button>
      </div>

      {browsing && (
        <FileBrowser
          onClose={() => setBrowsing(false)}
          onPick={(path) => {
            setSourcePath(path);
            setBrowsing(false);
          }}
        />
      )}
    </main>
  );
}
