#!/usr/bin/env bash
set -eo pipefail
ddla_root=${1:?run directory}
ddla_stage=${2:?build or case}
shift 2
source "$ddla_root/control/modules.sh"
unset UCX_TLS OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_coll LD_PRELOAD
export DDLA_ALLOCATED_GPU_MASK=${CUDA_VISIBLE_DEVICES:?}
ddla_tmp=$(mktemp -d "/tmp/libddla-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$ddla_tmp"' EXIT
export TMPDIR=$ddla_tmp
set -u
if [[ "$ddla_stage" == build ]]; then
    bash "$ddla_root/control/build.sh" "$ddla_root" "$@"
else
    python3 "$ddla_root/control/runtime.py" case "$ddla_root" "${1:?group}" "${SLURM_ARRAY_TASK_ID:?}"
fi
