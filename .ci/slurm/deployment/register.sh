#!/usr/bin/env bash
set -euo pipefail
# Read a fresh fork registration token from stdin; never persist or print it.
docker run --rm --interactive --log-driver none \
    --env ACTIONS_RUNNER_PRINT_LOG_TO_STDOUT=false \
    --user 1001:1001 --cap-drop ALL --security-opt no-new-privileges \
    --mount type=volume,src=libddla-daily-identity,dst=/home/runner \
    --workdir /home/runner \
    ghcr.io/actions/actions-runner@sha256:6d0b2d80d80f143f77dcb676e50920100bbb547e39e883454d57f45ab8c31300 bash -c '
      set -euo pipefail
      umask 077
      test ! -e .runner
      IFS= read -r ACTIONS_RUNNER_INPUT_TOKEN
      export ACTIONS_RUNNER_INPUT_TOKEN
      ./config.sh --unattended --url https://github.com/Stardust0831/LibDDLA \
        --name sai-libddla-daily-01 --labels sai-libddla-daily --work _work
      unset ACTIONS_RUNNER_INPUT_TOKEN
      chmod 600 .runner .credentials .credentials_rsaparams
      jq -e "(.ephemeral // false) == false" .runner >/dev/null
    '
