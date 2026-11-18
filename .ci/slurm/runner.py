#!/usr/bin/env python3
"""Trusted site relay: pin source, transfer, stream progress, collect and clean."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from runtime import config, write_json


HERE = Path(__file__).resolve().parent
MAX_ARCHIVE = 512 * 1024 * 1024


def extract(archive, target):
    total, seen = 0, set()
    with tarfile.open(archive) as stream:
        for member in stream:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or member.name in seen:
                raise ValueError('Unsafe or duplicate archive path')
            seen.add(member.name)
            if len(seen) > 30000:
                raise ValueError('Archive contains too many entries')
            destination = target / path
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                if total > MAX_ARCHIVE:
                    raise ValueError('Archive is too large')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('wb') as out:
                    shutil.copyfileobj(stream.extractfile(member), out)
                destination.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise ValueError('Archive links and special files are not accepted')


def prepare(args, cfg):
    stage = args.work / 'package'
    request_path = stage / 'request.json'
    if request_path.exists():
        request = json.loads(request_path.read_text())
        if request['source_sha'] != args.source_sha or request['control_sha'] != args.control_sha:
            raise ValueError('Run inputs changed during recovery')
        return args.work / 'package.tar.gz'
    args.work.mkdir(parents=True, exist_ok=True)
    bare = args.work / 'source.git'
    git = ['git', '-c', 'core.hooksPath=/dev/null', '-C', str(bare)]
    env = {**os.environ, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_TERMINAL_PROMPT': '0'}
    env.pop('GH_TOKEN', None)
    subprocess.run(['git', 'init', '--quiet', '--bare', str(bare)], check=True, env=env)
    subprocess.run(git + ['fetch', '--quiet', '--depth=1', '--no-tags',
                          'https://github.com/' + cfg['source']['repository'] + '.git', args.source_sha],
                   check=True, env=env, timeout=300)
    actual = subprocess.check_output(git + ['rev-parse', 'FETCH_HEAD^{commit}'], text=True, env=env).strip()
    if actual != args.source_sha:
        raise ValueError('Source fetch did not resolve the exact requested SHA')
    stage.mkdir(parents=True, exist_ok=True)
    (stage / 'results').mkdir(exist_ok=True)
    source_archive = stage / 'results/source.tar.gz'
    with source_archive.open('wb') as stream:
        subprocess.run(git + ['archive', '--format=tar.gz', actual], stdout=stream, check=True, env=env)
    extract(source_archive, stage / 'source')
    shutil.copytree(HERE, stage / 'control', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (stage / 'control').glob('*.sh'):
        path.chmod(0o755)
    with tarfile.open(stage / 'results/control.tar.gz', 'w:gz') as archive:
        archive.add(stage / 'control', arcname='control')
    request = {'run_id': args.run_id, 'source_sha': actual, 'control_sha': args.control_sha,
               'source_repository': cfg['source']['repository'], 'source_branch': cfg['source']['branch'],
               'created_at': time.time(), 'source_archive_sha256': hashlib.sha256(source_archive.read_bytes()).hexdigest(),
               'controller_files': {p.relative_to(stage / 'control').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((stage / 'control').rglob('*')) if p.is_file()}}
    write_json(request_path, request)
    write_json(stage / 'results/request.json', request)
    package = args.work / 'package.tar.gz'
    with tarfile.open(package, 'w:gz') as archive:
        for path in sorted(stage.iterdir()):
            archive.add(path, arcname=path.name)
    return package


class Remote:
    def __init__(self, ssh_config, cfg, run_id):
        self.base = ['ssh', '-F', str(ssh_config), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                     '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', cfg['remote']['host_alias']]
        self.root = Path(cfg['remote']['project_root']) / 'runs' / run_id

    def call(self, command, *, stdin=None, stdout=None, timeout=180):
        return subprocess.run(self.base + [command], stdin=stdin or subprocess.DEVNULL,
                              stdout=stdout or subprocess.PIPE, check=True, timeout=timeout).stdout

    def rpc(self, action, **kwargs):
        command = shlex.join(['python3', str(self.root / 'control/runner.py'), '--remote', action, str(self.root)])
        response = self.call(command, **kwargs)
        return json.loads(response) if response else None

    def upload(self, package):
        quoted = shlex.quote(str(self.root))
        staging = shlex.quote(str(self.root) + '.upload')
        # A successful retry consumes the upload without overwriting an active run.
        command = (f'if test -s {quoted}/request.json; then cat >/dev/null; else '
                   f'mkdir -p {staging}; tar -xzf - -C {staging} && mv {staging} {quoted}; fi')
        self.call(shlex.join(['mkdir', '-p', str(self.root.parent)]))
        with package.open('rb') as stream:
            self.call(command, stdin=stream, timeout=300)


def remote_action(action, root):
    if action in ('start', 'status'):
        (root / 'heartbeat').touch()
        if not (root / 'results/completion.json').exists():
            with (root / 'coordinator.log').open('a') as log:
                subprocess.Popen(['python3', '-u', str(root / 'control/coordinator.py'), str(root)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        progress = root / 'results/progress.log'
        completion = root / 'results/completion.json'
        return {'request': json.loads((root / 'request.json').read_text()),
                'progress': progress.read_text() if progress.exists() else '',
                'completion': json.loads(completion.read_text()) if completion.exists() else None}
    if action == 'cancel':
        from coordinator import cancel
        (root / 'cancel').touch()
        path = root / 'results/jobs.json'
        if path.exists():
            cancel(json.loads(path.read_text()))
        return {'cancel_requested': True}
    if action == 'archive':
        shutil.copy2(root / 'coordinator.log', root / 'results/coordinator.log')
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as stream:
            for path in sorted((root / 'results').rglob('*')):
                if path.is_file() and not path.is_symlink():
                    stream.add(path, arcname=path.relative_to(root / 'results'), recursive=False)
        return None
    if action == 'cleanup':
        from coordinator import done
        jobs = json.loads((root / 'results/jobs.json').read_text())
        if not (root / 'results/completion.json').exists() or not all(done(j) for j in jobs.values()):
            raise ValueError('Refusing to clean an unfinished Slurm run')
        for name in ('build', 'source'):
            shutil.rmtree(root / name, ignore_errors=True)
        # Retain raw evidence for 30 days; remove only completed, inactive runs.
        for old in root.parent.glob('gh-*-*'):
            receipt = old / 'results/completion.json'
            if old.is_dir() and receipt.is_file() and time.time() - receipt.stat().st_mtime > 30 * 86400:
                old_jobs = json.loads((old / 'results/jobs.json').read_text())
                if all(done(j) for j in old_jobs.values()):
                    shutil.rmtree(old)
        return {'cleaned': True}
    raise ValueError('Unknown remote operation')


def run(args):
    if not re.fullmatch(r'gh-[1-9][0-9]*-[1-9][0-9]*', args.run_id):
        raise ValueError('Invalid run identifier')
    for sha in (args.source_sha, args.control_sha):
        if not re.fullmatch(r'[a-f0-9]{40}', sha):
            raise ValueError('Full commit SHAs are required')
    cfg = config(HERE / 'config.ini')
    remote = Remote(args.ssh_config, cfg, args.run_id)
    args.results.mkdir(parents=True, exist_ok=True)
    errors, ready, complete, previous = [], False, False, ''
    started = time.monotonic()

    def emit(message):
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + message, flush=True)

    def interrupt(signum, frame):
        raise RuntimeError(f'Relay interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    try:
        emit('[1/5 prepare] Fetching upstream develop commit ' + args.source_sha)
        package = prepare(args, cfg)
        emit(f'[1/5 prepare] Uploading source and controller ({package.stat().st_size} bytes)')
        for attempt in range(3):
            try:
                remote.upload(package)
                ready = True
                break
            except subprocess.SubprocessError:
                if attempt == 2:
                    raise
                time.sleep(10)
        failures = 0
        while True:
            if time.monotonic() - started > cfg.getint('slurm', 'deadline_seconds') + 180:
                raise TimeoutError('Relay deadline reached')
            try:
                status = remote.rpc('status')
                failures = 0
            except subprocess.SubprocessError as error:
                failures += 1
                emit(f'[connection] Status retry {failures}/5: {type(error).__name__}')
                if failures >= 5:
                    raise
                time.sleep(20)
                continue
            if status['request']['source_sha'] != args.source_sha or status['request']['control_sha'] != args.control_sha:
                raise ValueError('Remote run has different source or controller inputs')
            progress = status['progress']
            if progress != previous:
                print(progress[len(previous):] if progress.startswith(previous) else progress, end='', flush=True)
                previous = progress
            if status['completion']:
                complete = True
                break
            time.sleep(cfg.getint('slurm', 'poll_seconds'))
    except Exception as error:
        errors.append(str(error))
        emit('[INFRA] ' + str(error))
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_DFL)
        if ready and not complete:
            try:
                remote.rpc('cancel')
                for _ in range(12):
                    if remote.rpc('status')['completion']:
                        break
                    time.sleep(5)
            except Exception as error:
                errors.append('Cancellation unconfirmed: ' + str(error))
        if ready:
            emit('[5/5 collect] Downloading raw logs, source archive and structured results')
            try:
                archive = args.work / 'results.tar.gz'
                for attempt in range(3):
                    try:
                        with archive.open('wb') as stream:
                            remote.rpc('archive', stdout=stream, timeout=300)
                        extract(archive, args.results / 'remote')
                        break
                    except (subprocess.SubprocessError, tarfile.TarError):
                        if attempt == 2:
                            raise
                        time.sleep(10)
                remote.rpc('cleanup')
                emit('[5/5 collect] Results downloaded; remote build/source removed; raw evidence retained 30 days')
            except Exception as error:
                errors.append('Collection or cleanup: ' + str(error))
        write_json(args.results / 'relay-errors.json', errors)
        summary = args.results / 'remote/summary.md'
        summary.parent.mkdir(exist_ok=True)
        if not summary.exists():
            summary.write_text('# LibDDLA daily GPU validation\n\nINFRA: no complete test report was produced.\n')
        if errors:
            with summary.open('a') as stream:
                stream.write('\n## Relay errors\n\n' + '\n'.join('- ' + e for e in errors) + '\n')
    result_path = args.results / 'remote/result.json'
    success = result_path.exists() and json.loads(result_path.read_text()).get('success', False)
    return 0 if success and not errors else 1


if __name__ == '__main__':
    if sys.argv[1:2] == ['--remote']:
        response = remote_action(sys.argv[2], Path(sys.argv[3]))
        if response is not None:
            print(json.dumps(response))
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--run-id', required=True)
        parser.add_argument('--source-sha', required=True)
        parser.add_argument('--control-sha', required=True)
        parser.add_argument('--ssh-config', required=True, type=Path)
        parser.add_argument('--work', required=True, type=Path)
        parser.add_argument('--results', required=True, type=Path)
        raise SystemExit(run(parser.parse_args()))
