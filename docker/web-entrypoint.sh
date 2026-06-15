#!/bin/sh
# Web container entrypoint.
#
# We intentionally use `/bin/sh` (not bash) so this works on any base
# image, including future moves to alpine / distroless variants.
#
# Order of operations:
#   1. Refuse to start if the three web auth secrets aren't set.
#      Better to fail loud here than to discover at request time that
#      auth is misconfigured.
#   2. Run the schema migration. This is idempotent (the migration
#      module skips already-applied versions) so it's safe to run on
#      every container start.
#   3. Exec into uvicorn so it inherits PID 2 (tini is PID 1) and
#      receives SIGTERM directly.

set -eu

if [ -z "${WEB_USERNAME:-}" ] || \
   [ -z "${WEB_PASSWORD:-}" ] || \
   [ -z "${WEB_SESSION_SECRET:-}" ]; then
  echo "FATAL: WEB_USERNAME / WEB_PASSWORD / WEB_SESSION_SECRET must be set." >&2
  echo "       Provide them via .env (mounted by docker-compose) or"            >&2
  echo "       --env when running docker directly. See .env.example."           >&2
  exit 78   # EX_CONFIG, in BSD sysexits.h convention
fi

# Idempotent schema migration. Logs go to stdout via the `loguru` logger.
python -m src.db.migrate

# `--proxy-headers` honours X-Forwarded-Proto from Cloudflare Tunnel so
# the security middleware can flip on HSTS + secure cookies once we're
# behind TLS. We only trust forwards from the docker-network gateway
# range; tightened further in Phase C.
exec python -m uvicorn src.web.app:create_app \
    --factory \
    --host "${WEB_BIND_HOST:-0.0.0.0}" \
    --port "${WEB_BIND_PORT:-8000}" \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips "${WEB_FORWARDED_IPS:-127.0.0.1,172.16.0.0/12}" \
    --no-access-log \
    --timeout-keep-alive 5
