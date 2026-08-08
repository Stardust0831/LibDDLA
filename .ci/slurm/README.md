# SAI GPU validation

The `GPU validation` workflow sends the selected committed LibDDLA source to
the SAI cluster, builds the CUDA backend, and runs every CTest entry through
Slurm. The GitHub job summary contains one table row per registered test. Raw
CTest, JUnit, build, module, GPU, and Slurm logs are retained as an artifact.

The current allocation uses one V100 for the build and eight V100s for the
test job. Tests run sequentially inside that allocation. This supports the
normal four-rank tests and the fixed six-rank `test_ptran_mpi` case without
oversubscribing GPUs.

## GitHub setup

The repository has two Actions environments:

- `gpu-ci-manual` for `workflow_dispatch` runs
- `gpu-ci-scheduled` for the daily scheduled run

Each environment requires a `REMOTE_SSH_PRIVATE_KEY` secret containing the
private key for the remote user configured in `config.ini`. Host verification
uses the committed `known_hosts` entry; do not disable it.

Manual and scheduled runs default to the `develop` branch. A manual run may
provide another branch name or full commit SHA through `source_sha`.

## Remote layout

Each run is created below:

```text
<project_root>/runs/github/<github_run_id>-<attempt>/
```

The `results/` directory contains `result.json`, `summary.md`, `ctest.xml`,
`ctest.log`, build logs, tool/module records, GPU details, and Slurm output.
The workflow downloads this directory even when tests fail.
