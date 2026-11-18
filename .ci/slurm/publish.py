#!/usr/bin/env python3
"""Site-side mailbox service. Install outside the Actions runner container."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

from runtime import config, write_json


REPO = 'Stardust0831/LibDDLA'
WORKFLOW = '.github/workflows/gpu-validation.yml'
LABEL = 'sai-libddla-daily'
HERE = Path(__file__).resolve().parent


def api(path, repo=REPO):
    return json.loads(subprocess.check_output(['gh', 'api', '--hostname', 'github.com', f'repos/{repo}/{path}'], timeout=45))


def admission(run, control_sha):
    if (run['repository']['full_name'] != REPO or run['head_repository']['full_name'] != REPO
            or run['path'] != WORKFLOW or run['event'] not in ('schedule', 'workflow_dispatch')
            or run['head_branch'] != 'main' or run['head_sha'] != control_sha):
        raise ValueError('Only the installed fork/main daily workflow is admitted')


def publish(run, result):
    number, attempt = run['id'], run['run_attempt']
    key = f'gh-{number}-{attempt}'
    control_sha = os.environ['LIBDDLA_CONTROL_SHA']
    work = Path(os.environ['SAI_WORK']) / key
    result.mkdir(exist_ok=True)
    code, sha, error = 1, None, None
    relay = None

    def interrupted(signum, frame):
        raise RuntimeError(f'Site service interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    try:
        admission(run, control_sha)
        cfg = config(HERE / 'config.ini')
        receipt = result / 'request.json'
        if receipt.exists():
            sha = json.loads(receipt.read_text())['source_sha']
        else:
            sha = api('commits/' + cfg['source']['branch'], repo=cfg['source']['repository'])['sha']
            if not re.fullmatch(r'[0-9a-f]{40}', sha):
                raise ValueError('Invalid upstream develop SHA')
            write_json(receipt, {'run_id': key, 'source_sha': sha, 'control_sha': control_sha})
        env = dict(os.environ)
        env.pop('GH_TOKEN', None)
        with (result / 'progress.log').open('a') as log:
            relay = subprocess.Popen([sys.executable, '-u', str(HERE / 'runner.py'), '--run-id', key,
                                      '--source-sha', sha, '--control-sha', control_sha,
                                      '--ssh-config', os.environ['SAI_SSH_CONFIG'], '--work', str(work),
                                      '--results', str(result)], stdout=log, stderr=subprocess.STDOUT, env=env)
            failures = 0
            while True:
                try:
                    code = relay.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        active = api(f'actions/runs/{number}')
                        failures = 0
                    except subprocess.SubprocessError:
                        failures += 1
                        if failures < 3:
                            continue
                        raise
                    if active['run_attempt'] != attempt or active['status'] != 'in_progress':
                        raise RuntimeError('Workflow ended or was superseded; cancelling its Slurm jobs')
    except Exception as exc:
        error = str(exc)
        print(f'{key}: {error}', file=sys.stderr, flush=True)
    finally:
        if relay is not None and relay.poll() is None:
            relay.send_signal(signal.SIGINT)
            try:
                relay.wait(timeout=240)
            except subprocess.TimeoutExpired:
                relay.kill()
                relay.wait()
                error = (error or '') + '; relay cleanup timed out (remote lease remains bounded)'
        if error:
            summary = result / 'remote/summary.md'
            summary.parent.mkdir(exist_ok=True)
            with summary.open('a') as stream:
                stream.write('\nSite service: INFRA — ' + error + '\n')
        write_json(result / 'completion.json', {'run_id': key, 'source_sha': sha, 'control_sha': control_sha,
                                               'exit_code': code, 'error': error})
        if work.is_dir():
            shutil.rmtree(work)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_DFL)


def main():
    public, private, mailbox = (Path(os.environ[key]).resolve(strict=True)
                                for key in ('SAI_RESULTS', 'SAI_WORK', 'SAI_REQUESTS'))
    if public in private.parents or private in public.parents or public == private:
        raise ValueError('Private work and public results must be separate')
    for request in sorted(mailbox.glob('gh-*-*.token')):
        match = re.fullmatch(r'gh-([1-9][0-9]*)-([1-9][0-9]*)\.token', request.name)
        if not match:
            continue
        result = public / request.stem
        descriptor = None
        try:
            if (result / 'completion.json').exists():
                request.unlink(missing_ok=True)
                continue
            descriptor = os.open(request, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError('Token must be a regular file')
            token = os.read(descriptor, 8193).decode('ascii')
            if len(token) > 8192 or not re.fullmatch(r'[!-~]+', token):
                raise ValueError('Invalid token format or size')
            os.environ['GH_TOKEN'] = token
            run = api('actions/runs/' + match[1])
            if run['run_attempt'] != int(match[2]) or run['status'] == 'completed':
                request.unlink(missing_ok=True)
                continue
            if run['status'] != 'in_progress':
                continue
            admission(run, os.environ['LIBDDLA_CONTROL_SHA'])
            jobs = api(f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs")['jobs']
            if any(job['status'] == 'in_progress' and LABEL in job['labels'] for job in jobs):
                publish(run, result)
        except Exception as error:
            print(f'{request.name}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            if isinstance(error, ValueError):
                result.mkdir(exist_ok=True)
                (result / 'remote').mkdir(exist_ok=True)
                (result / 'remote/summary.md').write_text('# LibDDLA daily GPU validation\n\nINFRA: ' + str(error) + '\n')
                write_json(result / 'completion.json', {'run_id': request.stem, 'exit_code': 1, 'error': str(error)})
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.environ.pop('GH_TOKEN', None)
            if (result / 'completion.json').exists():
                request.unlink(missing_ok=True)
    for old in public.glob('gh-*-*'):
        receipt = old / 'completion.json'
        if receipt.is_file() and time.time() - receipt.stat().st_mtime > 30 * 86400:
            shutil.rmtree(old)


if __name__ == '__main__':
    main()
