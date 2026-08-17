#!/usr/bin/env bash
set -euo pipefail

run_root=${1:?run root is required}
control=$run_root/control
source_dir=$run_root/source
build=$run_root/build
results=$run_root/results
mkdir -p "$control" "$build" "$results"

cfg() {
    python3 - "$control/config.ini" "$1" "$2" <<'PY'
import configparser
import sys
config = configparser.ConfigParser()
config.read(sys.argv[1])
print(config[sys.argv[2]][sys.argv[3]].strip())
PY
}

render() {
    python3 - "$@" <<'PY'
from pathlib import Path
import sys
template = Path(sys.argv[1]).read_text()
values = {
    "@BUILD_PARTITION@": sys.argv[3], "@BUILD_QOS@": sys.argv[4],
    "@BUILD_TIME@": sys.argv[5], "@TEST_PARTITION@": sys.argv[6],
    "@TEST_QOS@": sys.argv[7], "@TEST_TIME@": sys.argv[8],
    "@HOME@": sys.argv[9], "@CONTROL@": sys.argv[10],
    "@SOURCE@": sys.argv[11], "@BUILD@": sys.argv[12],
    "@RESULTS@": sys.argv[13], "@MAPPING_ROOT@": sys.argv[14],
    "@DISABLE_NCCL_IB@": sys.argv[15], "@BUILD_JOB@": sys.argv[16],
    "@CONTAINER_IMAGE@": sys.argv[17], "@MPI_ROOT@": sys.argv[18],
}
for key, value in values.items():
    template = template.replace(key, value)
Path(sys.argv[2]).write_text(template)
PY
}

build_partition=$(cfg build partition)
build_qos=$(cfg build qos)
build_time=$(cfg build time)
test_partition=$(cfg test partition)
test_qos=$(cfg test qos)
test_time=$(cfg test time)
mapping_root=$(cfg cluster mapping_root)
disable_nccl_ib=$(cfg cluster disable_nccl_ib)
mpi_root=$(cfg container mpi_root)
home_dir=$(getent passwd "$USER" | cut -d: -f6)
container_image=$(cfg container rootfs)
if [[ "$container_image" == "~/"* ]]; then
    container_image="$home_dir/${container_image#\~/}"
fi

build_script=$control/build.sbatch
render "$control/build.sbatch.in" "$build_script" \
    "$build_partition" "$build_qos" "$build_time" "$test_partition" \
    "$test_qos" "$test_time" "$home_dir" "$control" "$source_dir" \
    "$build" "$results" "$mapping_root" "$disable_nccl_ib" "0" \
    "$container_image" "$mpi_root"
chmod 700 "$build_script"
build_job=$(sbatch --parsable "$build_script")
echo "build_job=$build_job" | tee "$results/jobs.txt"

wait_job() {
    local job=$1
    while squeue -h -j "$job" | grep -q .; do
        squeue -h -j "$job" -o "%i %T %M %R" >&2 || true
        sleep 30
    done
    sacct -X -j "$job" --format=State,ExitCode -n -P | head -n 1
}

build_accounting=$(wait_job "$build_job")
echo "build_accounting=$build_accounting" | tee -a "$results/jobs.txt"
if [[ "$build_accounting" != COMPLETED\|0:0 ]]; then
    echo "build failed" >&2
    exit 1
fi

test_script=$control/test.sbatch
render "$control/test.sbatch.in" "$test_script" \
    "$build_partition" "$build_qos" "$build_time" "$test_partition" \
    "$test_qos" "$test_time" "$home_dir" "$control" "$source_dir" \
    "$build" "$results" "$mapping_root" "$disable_nccl_ib" "$build_job" \
    "$container_image" "$mpi_root"
chmod 700 "$test_script"
test_job=$(sbatch --parsable "$test_script")
echo "test_job=$test_job" | tee -a "$results/jobs.txt"
test_accounting=$(wait_job "$test_job")
echo "test_accounting=$test_accounting" | tee -a "$results/jobs.txt"
test -s "$results/result.json"
test -s "$results/summary.md"
[[ "$test_accounting" == COMPLETED\|0:0 ]]
