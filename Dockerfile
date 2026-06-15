# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the AI Trader.
#
# Stage 1 (builder): installs Python deps into a venv, including build
#                    deps that we DON'T want in the final image.
# Stage 2 (runtime): copies just the venv + app code, runs as a non-root
#                    user with no shell, drops privileges, opens only
#                    port 8000.
#
# Why slim-bookworm and not distroless / alpine:
# - Alpine uses musl libc, which is incompatible with the manylinux
#   wheels for numpy / scipy / xgboost. Building from source on ARM
#   would balloon the build time to 30+ minutes.
# - Distroless has no shell, which is a security win, but our
#   healthcheck uses `python -c` (not curl), and we still want a
#   minimal busybox-y environment for entrypoint shells.
#
# Why pin Python 3.11 specifically:
# - 3.11 has measurable speedups on the pandas/numpy hot paths we hit
#   in feature engineering (~10-20% on Apple-Silicon-class ARM cores).
# - 3.12+ broke a few transitive deps when we last tested; 3.11 is the
#   safe sweet spot for the requirements.txt we have today.
#
# Image is multi-arch: works on linux/amd64 (your laptop, most VPS) and
# linux/arm64 (Oracle Cloud Free Ampere). Build with:
#   docker buildx build --platform linux/amd64,linux/arm64 .
#
# TODO(security): pin the base image to a SHA digest before deploy. Get
# it once with `docker buildx imagetools inspect python:3.11-slim-bookworm`
# and replace the FROM line with `python@sha256:...`.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Build deps for any packages without prebuilt wheels on our target
# platforms. We deliberately install these only in the builder stage so
# the runtime image stays small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libffi-dev \
 && rm -rf /var/lib/apt/lists/*

# Create the venv inside the builder so we can copy it whole into the
# runtime image without dragging the apt build deps along.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Layer ordering: requirements.txt rarely changes -> cache the heaviest
# step. We use a BuildKit cache mount so re-builds reuse the wheel cache
# without persisting it into the layer.
WORKDIR /build
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel \
 && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}" \
    TZ=Asia/Kolkata \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Sane uvicorn defaults for being behind Cloudflare Tunnel; the
    # scheduler container ignores these.
    WEB_BIND_HOST=0.0.0.0 \
    WEB_BIND_PORT=8000

# tzdata: needed so APScheduler / our own scheduler see Asia/Kolkata.
# tini  : lightweight init that reaps zombies and forwards SIGTERM
#         (without it, Ctrl-C / docker stop takes 10s before SIGKILL).
# ca-certificates: HTTPS to yfinance / NSE / Cloudflare.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      tini \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
 && echo "${TZ}" > /etc/timezone

# Non-root user. UID 10001 is high enough to never collide with
# distro-managed users on the host. The home dir is read-only at runtime;
# all writes go to volume-mounted paths under /app/data, /app/logs, etc.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid ${APP_GID} aitrader \
 && useradd --system --uid ${APP_UID} --gid ${APP_GID} \
            --home-dir /app --shell /usr/sbin/nologin aitrader

# Bring in the venv built upstairs.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# App code. We copy explicit subdirs (not `COPY . .`) so a stray
# .env / .venv / data/ / models/ never sneaks into the image even if
# .dockerignore is misconfigured. Defence in depth.
COPY --chown=aitrader:aitrader src        ./src
COPY --chown=aitrader:aitrader scripts    ./scripts
COPY --chown=aitrader:aitrader docker     ./docker
COPY --chown=aitrader:aitrader pyproject.toml requirements.txt ./

# Volume mount points. We `mkdir` them here with the right ownership so
# bind-mounts and named volumes inherit the perms cleanly. Without this,
# a fresh `docker compose up` would create them as root on the host.
RUN mkdir -p /app/data /app/logs /app/reports /app/models \
 && chown -R aitrader:aitrader /app

# Make the entrypoints executable. Done as root before USER drop.
RUN chmod 0755 /app/docker/*.sh

USER aitrader

EXPOSE 8000

# Healthcheck hits /healthz which doesn't require auth. We keep the
# command pure-stdlib so we don't depend on curl/wget being installed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else 1)"

# tini as PID 1 -> proper signal handling. The default command is the
# web server; docker-compose overrides this for the scheduler service.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/web-entrypoint.sh"]
CMD []
