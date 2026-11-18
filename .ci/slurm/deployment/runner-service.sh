#!/usr/bin/env bash
set -euo pipefail
: "${LIBDDLA_SITE_ROOT:?}" "${LIBDDLA_RELEASE:?}"
ddla_image=ghcr.io/actions/actions-runner@sha256:6d0b2d80d80f143f77dcb676e50920100bbb547e39e883454d57f45ab8c31300
docker rm -f libddla-daily-runner >/dev/null 2>&1 || true
exec docker run --rm --name libddla-daily-runner --init --read-only \
    --user 1001:1001 --group-add "$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
    --memory 2g --cpus 2 --pids-limit 256 --tmpfs /tmp:rw,nosuid,nodev,size=256m \
    --env ACTIONS_RUNNER_PRINT_LOG_TO_STDOUT=false \
    --mount type=volume,src=libddla-daily-identity,dst=/home/runner \
    --mount "type=bind,src=$LIBDDLA_SITE_ROOT/results,dst=/srv/libddla-daily/results,readonly" \
    --mount "type=bind,src=$LIBDDLA_SITE_ROOT/requests,dst=/srv/libddla-daily/requests" \
    --mount "type=bind,src=$LIBDDLA_RELEASE,dst=/opt/libddla-control,readonly" \
    --mount "type=bind,src=$LIBDDLA_SITE_ROOT/bin/net-guard,dst=/opt/libddla-net-guard,readonly" \
    --workdir /home/runner "$ddla_image" /opt/libddla-net-guard /home/runner/run.sh
