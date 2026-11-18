#!/usr/bin/env python3
"""One resumable coordinator per run; build once, then one array per resource."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from runtime import config, report, write_json


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def clean_environment():
    # Do not inherit SBATCH_MEM*, SBATCH_PARTITION or other submission overrides.
    env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME') if k in os.environ}
    env.update(PATH='/usr/bin:/bin', LANG='C.UTF-8')
    return env


def capture(command, timeout=45):
    return subprocess.check_output(command, text=True, timeout=timeout, env=clean_environment()).strip()


def parse_accounting(text, names):
    tasks = {}
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) < 4 or not re.fullmatch(r'[0-9]+(?:_[0-9]+)?', fields[0]):
            continue
        job, name, state, exitcode = fields[:4]
        parent = job.split('_', 1)[0]
        if parent not in names or name != names[parent]:
            raise ValueError('Slurm job identity differs from the saved receipt')
        tasks[job] = {'state': state.split()[0].rstrip('+'), 'exit_code': exitcode}
    return tasks


def job_states(jobs):
    if not jobs:
        return {}
    names = {j['job_id']: j['name'] for j in jobs.values()}
    ids = ','.join(names)
    tasks = parse_accounting(capture(['sacct', '-X', '-n', '-P', '-j', ids,
                                      '--format=JobID%64,JobName%128,State%30,ExitCode']), names)
    for receipt in jobs.values():
        prefix = receipt['job_id']
        receipt['tasks'] = {key: value for key, value in tasks.items() if key == prefix or key.startswith(prefix + '_')}
    pending = ','.join(j['job_id'] for j in jobs.values() if not done(j))
    if not pending:
        return jobs
    # sacct can lag behind squeue and pending arrays can be compressed there.
    response = subprocess.run(['squeue', '-h', '-r', '-j', pending, '-o', '%i|%j|%T'],
                              capture_output=True, text=True, timeout=45, env=clean_environment())
    # Slurm removes finished jobs from squeue before accounting necessarily catches up.
    if response.returncode and 'Invalid job id specified' not in response.stderr:
        raise RuntimeError('squeue failed: ' + response.stderr.strip())
    queue = response.stdout
    for line in queue.splitlines():
        job, name, state = line.split('|')[:3]
        parent = job.split('_', 1)[0]
        if parent not in names or name != names[parent]:
            raise ValueError('Queued job identity differs from the saved receipt')
        if tasks.get(job, {}).get('state') not in TERMINAL:
            tasks[job] = {'state': state, 'exit_code': ''}
    for receipt in jobs.values():
        prefix = receipt['job_id']
        receipt['tasks'] = {key: value for key, value in tasks.items() if key == prefix or key.startswith(prefix + '_')}
    return jobs


def task_ids(receipt):
    if receipt['array_size']:
        return [receipt['job_id'] + '_' + str(i) for i in range(receipt['array_size'])]
    return [receipt['job_id']]


def done(receipt):
    return all(receipt.get('tasks', {}).get(key, {}).get('state') in TERMINAL for key in task_ids(receipt))


def cancel(jobs):
    # Both ID and exact name must match. Never cancel all jobs of the account.
    errors = []
    for receipt in jobs.values():
        if not done(receipt):
            response = subprocess.run(['scancel', '--name=' + receipt['name'], receipt['job_id']],
                                      env=clean_environment(), capture_output=True, text=True, timeout=45)
            # Completion may race cancellation. Continue cancelling the other arrays.
            if response.returncode and 'Invalid job id specified' not in response.stderr:
                errors.append(receipt['job_id'] + ': ' + response.stderr.strip())
    if errors:
        raise RuntimeError('; '.join(errors))


def submit(root, cfg, group, resource, count=0):
    request = json.loads((root / 'request.json').read_text())
    name = f"ddla-{request['run_id']}-{group}"
    intent = root / ('intent-' + group + '.json')
    if intent.exists():
        # A disconnected sbatch must be reconciled, never resubmitted blindly.
        accounting = capture(['sacct', '-X', '-n', '-P', '--name', name,
                               '--starttime', time.strftime('%Y-%m-%d', time.localtime(request['created_at'])),
                               '--format=JobID%64,JobName%128'])
        ids = {line.split('|')[0].split('_')[0] for line in accounting.splitlines()
               if len(line.split('|')) >= 2 and line.split('|')[1] == name}
        queued = capture(['squeue', '-h', '--name', name, '-o', '%A'])
        ids.update(queued.splitlines())
        if len(ids) != 1 or not re.fullmatch(r'[0-9]+', next(iter(ids), '')):
            raise RuntimeError(f'Cannot safely reconcile submission {name}: {sorted(ids)}')
        job_id = ids.pop()
    else:
        command = ['sbatch', '--parsable', '--job-name=' + name,
                   '--partition=' + cfg['slurm']['partition'], '--account=' + cfg['slurm']['account'],
                   '--qos=' + cfg['slurm']['qos'], '--nodes=1', '--ntasks=' + str(resource['ranks']),
                   '--ntasks-per-node=' + str(resource['ranks']), '--gpus-per-node=' + str(resource['gpus']),
                   '--time=' + str((resource['time_seconds'] + 59) // 60), '--export=ALL', '--chdir=' + str(root),
                   '--output=' + str(root / 'results/slurm' / (group + '-%A_%a.log'))]
        if count:
            command.append(f"--array=0-{count - 1}%{resource['parallelism']}")
        command += [str(root / 'control/job.sh'), str(root), 'case' if count else 'build',
                    group if count else str(resource['parallelism'])]
        write_json(intent, {'name': name, 'command': command})
        response = capture(command, timeout=90)
        job_id = response.split(';')[0]
        if not re.fullmatch(r'[0-9]+', job_id):
            raise RuntimeError('Invalid sbatch receipt: ' + response)
    return {'job_id': job_id, 'name': name, 'array_size': count, 'resource': resource, 'tasks': {}}


def run(root):
    cfg = config(root / 'control/config.ini')
    request = json.loads((root / 'request.json').read_text())
    out = root / 'results'
    (out / 'slurm').mkdir(parents=True, exist_ok=True)
    jobs_path = out / 'jobs.json'
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    errors, seen = [], set()
    previous = ''
    last_progress = 0

    def emit(phase, message):
        line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + f' [{phase}] ' + message
        with (out / 'progress.log').open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    def guard():
        if (root / 'cancel').exists():
            raise RuntimeError('Workflow cancelled')
        if time.time() - request['created_at'] > cfg.getint('slurm', 'deadline_seconds'):
            raise RuntimeError('Run deadline reached (including queue time)')
        if time.time() - (root / 'heartbeat').stat().st_mtime > cfg.getint('slurm', 'lease_seconds'):
            raise RuntimeError('Controller heartbeat expired')

    def refresh(phase):
        nonlocal previous, last_progress
        guard()
        job_states(jobs)
        write_json(jobs_path, jobs)
        parts = []
        for group, receipt in jobs.items():
            states = Counter(receipt['tasks'].get(key, {}).get('state', 'UNKNOWN') for key in task_ids(receipt))
            finished = sum(value for key, value in states.items() if key in TERMINAL)
            parts.append(f"{group} job={receipt['job_id']} done={finished}/{len(task_ids(receipt))} "
                         + ' '.join(f'{key}={value}' for key, value in sorted(states.items())))
        progress = '; '.join(parts)
        for path in sorted((out / 'cases').glob('*/result.json')):
            if path.parent.name not in seen:
                row = json.loads(path.read_text())
                emit('case', f"{row['id']} {row['name']} {row['state']} {row['duration']:.2f}s "
                            f"job={row['job_id']} log=cases/{row['id']}/ctest.log")
                seen.add(path.parent.name)
        if progress != previous or time.time() - last_progress >= 60:
            emit(phase, progress)
            previous, last_progress = progress, time.time()

    def interrupted(signum, frame):
        raise RuntimeError(f'Coordinator interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        emit('2/5 build', f"source={request['source_sha']} partition=16V100 qos=flood-gpu")
        if 'build' not in jobs:
            resource = {key: cfg.getint('build', key) for key in ('ranks', 'gpus', 'time_seconds', 'parallelism')}
            jobs['build'] = submit(root, cfg, 'build', resource)
            write_json(jobs_path, jobs)
        while True:
            refresh('2/5 build')
            if done(jobs['build']):
                break
            time.sleep(cfg.getint('slurm', 'poll_seconds'))
        if jobs['build']['tasks'].get(jobs['build']['job_id']) != {'state': 'COMPLETED', 'exit_code': '0:0'}:
            raise RuntimeError('Build failed; see build/ and slurm/build-*.log')
        plan = json.loads((out / 'plan.json').read_text())
        emit('3/5 arrays', f"{len(plan['cases'])} GPU cases: {plan['registered_gpu']} registered + "
                          f"{plan['extra_gpu']} extra; excluded={len(plan['excluded'])}")
        for group, resource in plan['groups'].items():
            guard()
            if group not in jobs:
                jobs[group] = submit(root, cfg, group, resource, len(resource['cases']))
                write_json(jobs_path, jobs)
                emit('3/5 arrays', f"submitted {group} job={jobs[group]['job_id']} "
                                  f"array=0-{len(resource['cases']) - 1}%{resource['parallelism']} "
                                  f"ranks={resource['ranks']} gpus={resource['gpus']}")
        while True:
            refresh('3/5 arrays')
            if all(done(receipt) for receipt in jobs.values()):
                break
            time.sleep(cfg.getint('slurm', 'poll_seconds'))
    except Exception as error:
        errors.append(str(error))
        emit('INFRA', str(error))
        try:
            cancel(jobs)
            for _ in range(12):
                job_states(jobs)
                if all(done(j) for j in jobs.values()):
                    break
                time.sleep(5)
            if not all(done(j) for j in jobs.values()):
                errors.append('Slurm cancellation has not been confirmed')
        except Exception as cleanup_error:
            errors.append('Cancellation: ' + str(cleanup_error))
    finally:
        write_json(jobs_path, jobs)
        emit('4/5 report', 'Collecting per-case results and Slurm accounting')
        result = report(root, jobs, errors)
        emit('4/5 report', ' '.join(f'{key}={value}' for key, value in result['counts'].items()))
        write_json(out / 'completion.json', {'run_id': request['run_id'], 'source_sha': request['source_sha'],
                                           'exit_code': 0 if result['success'] else 1})


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    with (root / 'coordinator.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(0)
        if not (root / 'results/completion.json').exists():
            run(root)
