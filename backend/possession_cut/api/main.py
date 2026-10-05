"""FastAPI application.

    uvicorn possession_cut.api.main:app

In development the React app is served by Vite on :5173 and proxies /api here. When
``frontend/dist`` exists (``npm run build``, or the Docker image) this app serves it too,
so one port is all there is.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .. import __version__
from ..config import REPO_ROOT, get_settings
from ..db import get_engine
from .routes import router
from .uploads import router as uploads_router

log = logging.getLogger(__name__)
DIST = REPO_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings().ensure_dirs()
    get_engine()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Possession Cut", version=__version__, lifespan=lifespan)
    cors: dict = {
        "allow_origins": settings.allowed_origins,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "max_age": 3600,
        "expose_headers": ["Content-Range", "Accept-Ranges", "Content-Length"],
    }
    if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
        # A hosted (https) copy of the page calling this local API is a public-to-private
        # request. Chrome asks for it by name on the preflight, and Starlette answers 400
        # unless told the listed origins may do that.
        cors["allow_private_network"] = True
    app.add_middleware(CORSMiddleware, **cors)

    @app.middleware("http")
    async def private_network_access(request: Request, call_next):
        """The same permission for a Starlette too old to know ``allow_private_network``."""
        response = await call_next(request)
        origin = request.headers.get("origin", "").rstrip("/")
        if origin in settings.allowed_origins and request.headers.get("access-control-request-private-network"):
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    app.include_router(router)
    app.include_router(uploads_router)

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):  # pragma: no cover - last resort
        log.exception("unhandled error")
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    if DIST.is_dir() and (DIST / "index.html").exists():
        index = DIST / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = (DIST / path).resolve()
            if path and candidate.is_file() and DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app


app = create_app()


def dist_path() -> Path:
    return DIST
