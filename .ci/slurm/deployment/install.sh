#!/usr/bin/env bash
set -euo pipefail
ddla_release=$(cd -- "$(dirname -- "$0")/.." && pwd)
ddla_sha=${1:?40-character controller commit}
ddla_ssh=${2:?path to a private SSH config with the configured SAI alias}
ddla_site=${3:-$HOME/.local/share/libddla-daily}
[[ "$ddla_sha" =~ ^[a-f0-9]{40}$ ]]
test -r "$ddla_ssh"
test -x "$HOME/.local/share/libddla-ci/bin/gh"
umask 022
mkdir -p "$ddla_site"/{bin,work,requests,results,gh-empty} "$HOME/.config/systemd/user"
chmod 700 "$ddla_site" "$ddla_site/work" "$ddla_site/gh-empty"
chmod 2770 "$ddla_site/requests"
printf '%s\n' "$ddla_sha" > "$ddla_release/control-sha"
gcc -O2 -Wall -Wextra -Werror "$ddla_release/deployment/net-guard.c" -o "$ddla_site/bin/net-guard"
chmod 755 "$ddla_release/"*.sh "$ddla_release/deployment/"*.sh "$ddla_site/bin/net-guard"
cat > "$HOME/.config/systemd/user/libddla-daily-publisher.service" <<EOF
[Unit]
Description=LibDDLA fork daily GPU site service
[Service]
Type=oneshot
Environment=PATH=$HOME/.local/share/libddla-ci/bin:/usr/bin:/bin
Environment=GH_CONFIG_DIR=$ddla_site/gh-empty
Environment=LIBDDLA_CONTROL_SHA=$ddla_sha
Environment=SAI_SSH_CONFIG=$ddla_ssh
Environment=SAI_WORK=$ddla_site/work
Environment=SAI_RESULTS=$ddla_site/results
Environment=SAI_REQUESTS=$ddla_site/requests
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=/usr/bin/python3 -u $ddla_release/publish.py
UMask=0022
NoNewPrivileges=yes
KillMode=mixed
KillSignal=SIGINT
TimeoutStartSec=infinity
TimeoutStopSec=300
MemoryMax=2G
CPUQuota=200%
EOF
cat > "$HOME/.config/systemd/user/libddla-daily-publisher.timer" <<'EOF'
[Unit]
Description=Poll LibDDLA fork daily GPU requests
[Timer]
OnBootSec=20
OnUnitInactiveSec=15
AccuracySec=1
[Install]
WantedBy=timers.target
EOF
cat > "$HOME/.config/systemd/user/libddla-daily-runner.service" <<EOF
[Unit]
Description=Persistent LibDDLA fork daily Actions runner
[Service]
Type=simple
Environment=LIBDDLA_SITE_ROOT=$ddla_site
Environment=LIBDDLA_RELEASE=$ddla_release
ExecStart=$ddla_release/deployment/runner-service.sh
ExecStop=/usr/bin/docker stop --time 60 libddla-daily-runner
Restart=always
RestartSec=15
TimeoutStopSec=90
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now libddla-daily-publisher.timer
# Register before starting the runner service. Reinstalling preserves its identity.
printf 'Installed %s. Register if needed, then start libddla-daily-runner.service.\n' "$ddla_sha"
