#!/bin/sh
# Scheduler container entrypoint.
#
# This container does NOT need the WEB_* secrets. It only needs the
# DB / data volume to be mounted so it can write today's run.
#
# We still run the migration here for two reasons:
#   1. If the scheduler comes up first (docker-compose start order is
#      not strictly guaranteed without `depends_on: condition`), the DB
#      file gets created and migrated before the daily run touches it.
#   2. Schema upgrades shouldn't require restarting the web container.

set -eu

python -m src.db.migrate

exec python -m scripts.run_scheduler "$@"
