# Possession Cut in one container: API, worker and the built frontend, with ffmpeg bundled.
#   docker compose up --build   ->   http://localhost:8000

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    INBOX_DIR=/app/inbox \
    DATA_DIR=/app/data \
    EXPORTS_DIR=/app/exports \
    ALLOWED_ROOTS=/games
# ffmpeg for decoding and rendering; libgl1/libglib2 for OpenCV; a bold sans font for titles
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/possession_cut backend/possession_cut
RUN pip install ./backend && pip install -e ./backend --no-deps
COPY --from=web /web/dist frontend/dist
COPY tools tools
RUN mkdir -p inbox data exports /games
EXPOSE 8000
# the launcher runs the API (which also serves the built frontend) and the worker together
CMD ["python", "-m", "possession_cut.dev", "--no-frontend"]
