#!/usr/bin/env python3
"""Actions-side client. No SSH key or general site credential is available here."""
import json
import os
from pathlib import Path
import shutil
import sys
import time


def collect_results(result, target, summary_path):
    target.mkdir(parents=True, exist_ok=True)
    if result.exists():
        shutil.copytree(result, target, dirs_exist_ok=True)
    summary = target / 'remote/summary.md'
    if not summary.exists():
        summary.parent.mkdir(exist_ok=True)
        summary.write_text('# LibDDLA daily GPU validation\n\nINCOMPLETE: no completed site report. '
                           'The run may have been cancelled before collection; available progress is in the artifact.\n')
    with summary_path.open('a') as stream:
        stream.write(summary.read_text())


def main():
    key = f"gh-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    request = Path('/srv/libddla-daily/requests') / (key + '.token')
    result = Path('/srv/libddla-daily/results') / key
    temporary = request.with_suffix('.tmp')
    token = os.environ.pop('GH_TOKEN')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(token)
    temporary.replace(request)
    del token
    offset = 0
    deadline = time.monotonic() + 15000
    code = 1
    try:
        print('[request] Waiting for site service to resolve upstream develop and start GPU validation', flush=True)
        while time.monotonic() < deadline:
            log = result / 'progress.log'
            if log.exists():
                with log.open() as stream:
                    stream.seek(offset)
                    print(stream.read(), end='', flush=True)
                    offset = stream.tell()
            completion = result / 'completion.json'
            if completion.exists():
                receipt = json.loads(completion.read_text())
                if receipt['run_id'] != key:
                    raise ValueError('Result belongs to another run')
                code = receipt['exit_code']
                break
            time.sleep(10)
        else:
            raise TimeoutError('Site service did not finish within the workflow deadline')
    finally:
        request.unlink(missing_ok=True)
    return code


if __name__ == '__main__':
    if sys.argv[1:] == ['--collect']:
        key = f"gh-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        collect_results(Path('/srv/libddla-daily/results') / key,
                        Path(os.environ['RUNNER_TEMP']) / 'libddla-gpu-results',
                        Path(os.environ['GITHUB_STEP_SUMMARY']))
    else:
        raise SystemExit(main())
