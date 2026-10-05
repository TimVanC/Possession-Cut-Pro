"""Start the API, the worker and the frontend together. Ctrl+C stops all three.

    python -m possession_cut.dev                 # dev servers (Vite on :5173, API on :8000)
    python -m possession_cut.dev --no-frontend   # the engine alone: API and worker
    python -m possession_cut.dev --no-frontend --background   # same, logging to data/engine.log
    python -m possession_cut.dev --stop          # stop a copy started with --background
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from .config import REPO_ROOT, ffmpeg_bin, get_settings

COLORS = {"api": "\033[36m", "worker": "\033[35m", "web": "\033[32m"}
RESET = "\033[0m"


def _pump(name: str, proc: subprocess.Popen) -> None:
    color = COLORS.get(name, "")
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            print(f"{color}[{name:>6}]{RESET} {line}", flush=True)


def _spawn(name: str, cmd: list[str], cwd: str, env: dict[str, str]) -> subprocess.Popen:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )
    threading.Thread(target=_pump, args=(name, proc), daemon=True).start()
    return proc


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _stop_background(pid_file) -> int:
    """Stop the copy that wrote ``pid_file`` (started with --background)."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        print("No background engine is recorded as running.")
        return 0
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    pid_file.unlink(missing_ok=True)
    print("Possession Cut engine stopped.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-frontend", action="store_true", help="skip the Vite dev server")
    parser.add_argument("--no-worker", action="store_true", help="skip the worker process")
    parser.add_argument("--reload", action="store_true", help="uvicorn auto-reload")
    parser.add_argument("--background", action="store_true", help="log to data/engine.log instead of the console")
    parser.add_argument("--stop", action="store_true", help="stop a copy started with --background")
    args = parser.parse_args()

    # Child processes print characters a Windows console code page cannot encode
    # (Vite's arrows); never let that kill the log pump.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    settings = get_settings()
    settings.ensure_dirs()
    pid_file = settings.data_path / "engine.pid"
    if args.stop:
        return _stop_background(pid_file)
    if _port_in_use(settings.api_host, settings.api_port):
        # started twice (a login item and a double-click): the first copy is the engine
        print(f"Possession Cut is already running on http://{settings.api_host}:{settings.api_port}.")
        return 0
    if args.background:
        log_path = settings.data_path / "engine.log"
        if log_path.exists() and log_path.stat().st_size > 5_000_000:
            log_path.unlink()
        sys.stdout = sys.stderr = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
        pid_file.write_text(str(os.getpid()))
    try:
        ffmpeg_bin()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not settings.anthropic_api_key:
        print(
            "NOTE: ANTHROPIC_API_KEY is not set. Calibration will use the local detector "
            "and captions a template. Add the key to .env for Claude vision."
        )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    backend = str(REPO_ROOT / "backend")
    procs: dict[str, subprocess.Popen] = {}

    api_cmd = [
        sys.executable, "-m", "uvicorn", "possession_cut.api.main:app",
        "--host", settings.api_host, "--port", str(settings.api_port),
    ]
    if args.reload:
        api_cmd.append("--reload")
    procs["api"] = _spawn("api", api_cmd, backend, env)

    if not args.no_worker:
        procs["worker"] = _spawn("worker", [sys.executable, "-m", "possession_cut.worker"], backend, env)

    web_url = f"http://{settings.api_host}:{settings.api_port}"
    if not args.no_frontend:
        npm = shutil.which("npm")
        frontend = REPO_ROOT / "frontend"
        if npm and (frontend / "package.json").exists():
            env_web = dict(env)
            env_web["VITE_API_TARGET"] = f"http://{settings.api_host}:{settings.api_port}"
            procs["web"] = _spawn("web", [npm, "run", "dev"], str(frontend), env_web)
            web_url = "http://localhost:5173"
        else:
            print("NOTE: npm or frontend/ not found; serving the built frontend from the API if present.")

    hosted = [o for o in settings.allowed_origins if o.startswith("https://")]
    if args.no_frontend and hosted:
        web_url = f"{hosted[0]}  (or {web_url})"
    print(f"\n  Possession Cut is starting. Open {web_url}\n  Press Ctrl+C to stop.\n", flush=True)

    code = 0
    try:
        while True:
            for name, proc in procs.items():
                rc = proc.poll()
                if rc is not None:
                    print(f"[{name}] exited with code {rc}; shutting down.", flush=True)
                    code = rc or 1
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs.values():
            _kill_tree(proc)
        if args.background:
            pid_file.unlink(missing_ok=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
