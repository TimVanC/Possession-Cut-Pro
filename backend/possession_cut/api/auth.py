"""One password in front of the app, for a copy that runs on a server.

Off unless ``APP_PASSWORD`` is set or the copy is ``HOSTED``. When on, every ``/api``
route except the health check and sign-in needs the session cookie that signing in sets.
A hosted copy with no password fails closed: it serves nothing until one is set.

The page and the API share an origin on a server, so the session is an HttpOnly cookie
the browser sends by itself with every request, video and event stream included.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import Settings, get_settings

router = APIRouter(prefix="/api")

COOKIE = "pc_session"
SESSION_SECONDS = 30 * 24 * 3600
OPEN_PATHS = {"/api/health", "/api/login", "/api/logout"}
MAX_FAILURES, FAILURE_WINDOW = 8, 600.0  # wrong passwords allowed per address per 10 minutes

_failures: dict[str, deque[float]] = defaultdict(deque)


def _signing_key(settings: Settings) -> bytes:
    """Derived from the password and a random value kept with the data, so a changed
    password signs everyone out and a token from one copy is no use on another."""
    path = settings.data_path / "session.key"
    try:
        salt = path.read_text(encoding="utf-8").strip()
    except OSError:
        salt = ""
    if len(salt) < 32:
        salt = os.urandom(32).hex()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(salt, encoding="utf-8")
    return hashlib.sha256(f"{salt}:{settings.app_password}".encode()).digest()


def make_token(settings: Settings, now: float | None = None) -> str:
    expires = int((now or time.time()) + SESSION_SECONDS)
    signature = hmac.new(_signing_key(settings), str(expires).encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def valid_token(settings: Settings, token: str | None, now: float | None = None) -> bool:
    if not token or not settings.app_password:
        return False
    expires, _, signature = token.partition(".")
    if not expires.isdigit() or int(expires) < (now or time.time()):
        return False
    expected = hmac.new(_signing_key(settings), expires.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def signed_in(request: Request, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return (not settings.auth_required) or valid_token(settings, request.cookies.get(COOKIE))


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _secure(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


NO_PASSWORD = (
    "This copy of Possession Cut has no password yet. Set APP_PASSWORD in the server's "
    "variables; it restarts by itself and you can sign in."
)


async def require_session(request: Request, call_next):
    """Middleware: turn away any API call that is not signed in."""
    settings = get_settings()
    path = request.url.path
    if not settings.auth_required or not path.startswith("/api/") or path in OPEN_PATHS or request.method == "OPTIONS":
        return await call_next(request)
    if not settings.app_password:
        return JSONResponse({"detail": NO_PASSWORD}, status_code=503)
    if not valid_token(settings, request.cookies.get(COOKIE)):
        return JSONResponse({"detail": "Sign in to continue."}, status_code=401)
    return await call_next(request)


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=500)


@router.post("/login")
def login(body: Login, request: Request, response: Response) -> dict:
    settings = get_settings()
    if not settings.auth_required:
        return {"authenticated": True}
    if not settings.app_password:
        raise HTTPException(503, NO_PASSWORD)
    now = time.time()
    recent = _failures[_client(request)]
    while recent and now - recent[0] > FAILURE_WINDOW:
        recent.popleft()
    if len(recent) >= MAX_FAILURES:
        raise HTTPException(429, "Too many wrong passwords. Wait ten minutes and try again.")
    given = hashlib.sha256(body.password.encode()).digest()
    wanted = hashlib.sha256(settings.app_password.encode()).digest()
    if not hmac.compare_digest(given, wanted):
        recent.append(now)
        raise HTTPException(401, "That password is not right.")
    recent.clear()
    response.set_cookie(
        COOKIE, make_token(settings), max_age=SESSION_SECONDS, httponly=True, samesite="lax",
        secure=_secure(request), path="/",
    )
    return {"authenticated": True}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE, path="/")
    return {"authenticated": False}
