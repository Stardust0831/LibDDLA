# Daily GPU validation on SAI

This fork tests `userchba/LibDDLA:develop` every day at **20:30 UTC / 04:30
Asia/Shanghai**, and supports manual dispatch from **fork `main`**. The site
service resolves upstream `develop` to a full commit SHA when a run starts.
The workflow controller and tested source have separate recorded SHAs. No
upstream repository settings or workflows are required.

Only tests labelled `gpu` in the CUDA build's CTest inventory are selected.
Two additional six-rank, 1×6-grid cases exercise `test_pgeadd_indexed` and
`test_pgemm_branches` using the same binaries. CPU-only and unlabelled entries
are listed as excluded. Counts are generated from each build, not hardcoded.
An empty inventory, unsupported CTest property or unconfigured MPI rank count
fails visibly so newly added tests cannot silently disappear.

## Resources and configuration

All build and test jobs use **`16V100`**, **`flood-gpu`**, account
`org-open-sai`, with **no explicit memory or CPU-count request**. This QoS
requires GPU allocations in multiples of four. Each MPI rank sees one distinct
GPU; unused allocated GPUs are recorded separately from the MPI rank count.

| Stage/group | MPI ranks | Allocated GPUs/task | Array concurrency | Task wall time |
|---|---:|---:|---:|---:|
| Build | 4 | 4 | 1 | 30 min |
| mpi1 | 1 | 4 | 2 | 15 min |
| mpi2 | 2 | 4 | 1 | 15 min |
| mpi4 | 4 | 4 | 4 | 15 min |
| mpi6 | 6 | 8 | 1 | 20 min |

A single build is followed by one Slurm array for each resource group. A
separate array element runs each test. Arrays may run concurrently, with the
limits above allowing at most 36 allocated GPUs in total. Slurm decides when
tasks start. The coordinator includes queue time in its four-hour deadline.

`config.ini` controls source selection, resource groups, time limits and array
concurrency. `modules.sh` pins CUDA 12.9.1, GCC 13.3.0, Open MPI 5.0.10,
NCCL 2.29.3 and the SAI BLAS dependency; the build targets V100 (`sm_70`).
`build.sh` records and checks linked CUDA/NCCL libraries. `mpiexec.sh` and
`rank.sh` implement rank-to-GPU mapping, which is checked by a CUDA probe in
every array element.

CTest commands, environment variables (including allocation tracing), timeout,
skip and failure properties are preserved. Each task has an isolated CTest
directory and a private copy of its original working directory, so concurrent
tests cannot overwrite one another's CTest files or relative outputs.

## Progress and results

Actions streams the site progress log with explicit phases:

1. Resolve source, prepare and upload the pinned source/controller.
2. Submit build and report pending/running/completed state.
3. Submit arrays; report resource-group progress and each completed case.
4. Reconcile results with Slurm accounting and generate the report.
5. Download evidence and remove remote build/source directories.

Each completed case log line identifies its number, name, state, duration,
Slurm `array_job_task` ID and artifact log path. The Actions summary includes
source/controller SHAs, totals, build/array IDs, unsuccessful cases and a full
per-case table. States are **PASS / FAIL / INFRA / SKIPPED**. A skipped or
missing test does not count as a pass. A passing CTest result also requires a
successful Slurm task completion. Any failure, infrastructure error or skip
makes the workflow fail; known failures are not suppressed.

The 30-day artifact contains `remote/summary.md`, `remote/result.json`, a
combined `remote/ctest.xml`, per-case JUnit/logs/GPU mappings, build and Slurm
logs, inventory and plan, and compressed source/controller inputs. The exact
source SHA and source archive checksum are in `remote/request.json`.

The previously observed two Cholesky non-positive-definite error-code failures
on CUDA 12.9.1 remain ordinary test failures. A completed CI pipeline does not
imply those numerical/API checks passed.

## Runner and site setup

The self-hosted runner is registered **only to `Stardust0831/LibDDLA`**, with
label `sai-libddla-daily`. The workflow has no PR trigger, activation variable
or Environment approval dependency. GitHub's daily schedule dispatches to the
runner; `runner_check=true` is a manual connectivity check that submits no
Slurm jobs.

The runner container has a persistent GitHub registration and limited
read/write mounts for a request mailbox and its own work directory. Results
and installed controller files are read-only mounts. It has no SAI SSH key,
Docker socket or host home mount. The network guard allows outbound TCP 443.
The job token has only Contents/Actions read permissions and is removed from
the mailbox after use; no personal GitHub token is needed by the site service.

A separate site service owns the dedicated SAI account's SSH configuration.
It accepts only this fork's `main` schedule/manual workflow, at the exact
installed controller commit, and checks that the requesting job is active.
It selects upstream `develop` itself. Native CMake/test code executes as the
dedicated SAI account, so this setup assumes upstream `develop` is trusted;
it is not a sandbox for arbitrary PR code.

For installation on the runner host (Docker, user systemd, Python 3, Git, GCC
and `gh` are required):

1. Copy `.ci/slurm` from the reviewed fork `main` commit to a versioned directory
   such as `~/.local/share/libddla-daily/releases/<commit>/`. Keep the SAI SSH
   private key outside this directory and use strict host-key verification.
2. Run `deployment/install.sh <full-controller-sha> <private-ssh-config>` from
   that copy. The SSH config must provide `SAI-libddla-ci`, matching
   `config.ini`. The installer reuses the site's existing `gh` executable at
   `~/.local/share/libddla-ci/bin/gh`; no saved authentication is used.
3. As a fork administrator, generate a fresh registration token for the fork,
   pipe it into `deployment/register.sh` on the host, then run
   `systemctl --user enable --now libddla-daily-runner.service`. Do not register
   an ephemeral runner or start it with `--once`.
4. Ensure user lingering is enabled so both services survive logout. Disable
   the retired upstream site's publisher/runner services locally. No upstream
   API operation is needed.
5. Dispatch **Daily GPU validation** on fork `main`, inspect the artifact and
   summary, then dispatch with `runner_check=true` to verify the registration
   remains usable after a job.

For controller changes, install the matching new fork `main` commit and
restart `libddla-daily-runner.service`. The site deliberately refuses a
workflow commit that differs from the installed controller. Registration is
preserved across controller updates.

On cancellation the site relay cancels only its recorded, name-matched Slurm
jobs. A lost relay heartbeat cancels the remote run within ten minutes, and
fixed job wall times and the run deadline also bound execution. Successful
collection removes build/source directories; completed raw evidence is kept
on the site and runner host for 30 days. No cleanup runs against active jobs.

Local checks: `python3 -m unittest discover -s .ci/slurm -p 'test_*.py'`.
