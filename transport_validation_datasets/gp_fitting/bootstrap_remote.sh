#!/usr/bin/env bash
# One-time setup of the GP fitting environment on a SLURM cluster.
#
# The fitting jobs need only python + numpy + scipy + mkgp, so instead of
# replicating the full repo environment remotely, this builds a minimal venv
# on the cluster scratch space. Pass the venv path it prints to
# ClusterFitConfig.venv_path.
#
# Usage:
#   bash bootstrap_remote.sh <ssh-host> <remote-workdir> [python-version]
#
#   ssh-host        Host alias from ~/.ssh/config (pass the same alias to
#                   ClusterFitConfig.ssh_host)
#   remote-workdir  Cluster scratch dir for fitting files, e.g. /pool001/$USER/gpfit
#   python-version  Optional, default 3.12

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ssh-host> <remote-workdir> [python-version]" >&2
    exit 1
fi

HOST="$1"
WORKDIR="$2"
PYVER="${3:-3.12}"

# Exact pins, kept in sync with uv.lock: the zk worker relies on
# non-public mkgp surfaces (see zk/kernel.py), and the fits only match
# a local run's when numpy and scipy are the same builds.
MKGP_SPEC="mkgp==3.1.4"
NUMPY_SPEC="numpy==1.26.4"
SCIPY_SPEC="scipy==1.17.1"

echo "==> Creating $WORKDIR on $HOST"
# Only the workdir itself: each dataset makes its own <workdir>/<ds_name>/
# subdirectory (batch files, job scripts, logs) at dispatch time.
# shellcheck disable=SC2029  # client-side expansion of WORKDIR is intended
ssh "$HOST" "mkdir -p '$WORKDIR'"

echo "==> Building venv with python $PYVER (installs uv if missing)"
# shellcheck disable=SC2087  # client-side expansion into the heredoc is
# intended (WORKDIR/PYVER/*_SPEC), remote-side variables are escaped
ssh "$HOST" bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd '$WORKDIR'
uv venv --python '$PYVER' .venv
uv pip install --python .venv/bin/python '$NUMPY_SPEC' '$SCIPY_SPEC' '$MKGP_SPEC'
.venv/bin/python -c "import mkgp; print('mkgp OK:', mkgp.__file__)"
EOF

echo
echo "Remote environment ready."
echo "  venv: $WORKDIR/.venv"
echo "  packages: $NUMPY_SPEC $SCIPY_SPEC $MKGP_SPEC (from PyPI)"
echo
echo "Pass to ClusterFitConfig:"
echo "  ssh_host='$HOST', remote_workdir='$WORKDIR', venv_path='$WORKDIR/.venv'"
