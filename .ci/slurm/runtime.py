#!/usr/bin/env python3
"""GPU inventory, isolated CTest execution and reports (runs inside Slurm)."""
import argparse
from collections import Counter
import configparser
import copy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


STATES = ('PASS', 'FAIL', 'INFRA', 'SKIPPED')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def config(path):
    result = configparser.ConfigParser(interpolation=None)
    if not result.read(path):
        raise ValueError('Missing config.ini')
    if result['slurm']['partition'] != '16V100' or result['slurm']['qos'] != 'flood-gpu':
        raise ValueError('This controller requires 16V100 / flood-gpu')
    return result


def properties(test):
    return {p['name']: p['value'] for p in test.get('properties', [])}


def mpi_ranks(command, launcher):
    if command[0] != str(launcher) or command[1] not in ('-n', '-np'):
        raise ValueError('GPU test does not use the configured MPI launcher: ' + shlex.join(command))
    ranks = int(command[2])
    if not 1 <= ranks <= 16:
        raise ValueError('Unsupported MPI rank count')
    return ranks


def make_plan(inventory, cfg, root):
    launcher = root / 'control/mpiexec.sh'
    groups, by_rank = {}, {}
    for section in cfg.sections():
        if not section.startswith('resource.'):
            continue
        name = section.split('.', 1)[1]
        if not re.fullmatch(r'[a-z0-9]+', name):
            raise ValueError('Invalid resource name')
        resource = {key: cfg.getint(section, key) for key in ('ranks', 'gpus', 'time_seconds', 'parallelism')}
        if resource['gpus'] % 4 or not 4 <= resource['gpus'] <= 16 or not 1 <= resource['ranks'] <= resource['gpus']:
            raise ValueError('GPU resources must satisfy the partition policy')
        if not 1 <= resource['parallelism'] <= 16 or not 120 <= resource['time_seconds'] <= 14400:
            raise ValueError('Invalid array limits')
        if resource['ranks'] in by_rank:
            raise ValueError('Ambiguous resources for an MPI rank count')
        groups[name] = {**resource, 'cases': []}
        by_rank[resource['ranks']] = name
    tests = [copy.deepcopy(t) for t in inventory['tests'] if 'gpu' in properties(t).get('LABELS', [])]
    registered = len(tests)
    if not tests:
        raise ValueError('CUDA build registered no GPU tests')
    for section in cfg.sections():
        if section.startswith('extra.'):
            name = section.split('.', 1)[1]
            binary = root / 'build/tests' / cfg[section]['binary']
            if not binary.is_file():
                raise ValueError('Missing extra GPU test binary: ' + str(binary))
            tests.append({'name': name,
                          'command': [str(launcher), '-n', cfg[section]['ranks'], str(binary),
                                      *shlex.split(cfg[section]['arguments'])],
                          'properties': [{'name': 'LABELS', 'value': ['gpu']},
                                         {'name': 'WORKING_DIRECTORY', 'value': str(binary.parent)},
                                         {'name': 'ENVIRONMENT', 'value': ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1']}]})
    names = set()
    cases = []
    supported = {'ENVIRONMENT', 'LABELS', 'PROCESSORS', 'SKIP_RETURN_CODE', 'TIMEOUT',
                 'WORKING_DIRECTORY', 'WILL_FAIL', 'PASS_REGULAR_EXPRESSION',
                 'FAIL_REGULAR_EXPRESSION', 'SKIP_REGULAR_EXPRESSION'}
    for test in tests:
        if test['name'] in names or not re.fullmatch(r'[A-Za-z0-9_.+-]+', test['name']):
            raise ValueError('Duplicate or invalid test name')
        names.add(test['name'])
        props = properties(test)
        if set(props) - supported:
            raise ValueError(f"Unsupported CTest properties for {test['name']}: {set(props) - supported}")
        ranks = mpi_ranks(test['command'], launcher)
        if ranks not in by_rank:
            raise ValueError(f'No resource group configured for {ranks} MPI ranks')
        group = by_rank[ranks]
        timeout = float(props.get('TIMEOUT', 600))
        if not 0 < timeout <= groups[group]['time_seconds'] - 120:
            raise ValueError(f"CTest timeout exceeds allocation for {test['name']}")
        case = {**test, 'id': f'{len(cases) + 1:03d}', 'group': group, 'ranks': ranks,
                'array_index': len(groups[group]['cases']), 'timeout': timeout,
                'origin': 'ctest' if len(cases) < registered else 'extra'}
        groups[group]['cases'].append(case['id'])
        cases.append(case)
    return {'registered_gpu': registered, 'extra_gpu': len(tests) - registered,
            'excluded': [t['name'] for t in inventory['tests'] if 'gpu' not in properties(t).get('LABELS', [])],
            'groups': {name: value for name, value in groups.items() if value['cases']}, 'cases': cases}


def cmake_quote(value):
    if isinstance(value, list):
        value = ';'.join(str(v) for v in value)
    elif isinstance(value, bool):
        value = 'TRUE' if value else 'FALSE'
    value = str(value)
    equals = '='
    while ']' + equals + ']' in value:
        equals += '='
    return '[' + equals + '[' + value + ']' + equals + ']'


def test_file(case, work):
    props = properties(case)
    props['WORKING_DIRECTORY'] = str(work)
    props['TIMEOUT'] = case['timeout']
    lines = ['add_test(' + ' '.join(cmake_quote(x) for x in [case['name'], *case['command']]) + ')']
    for key, value in props.items():
        lines.append(f"set_tests_properties({cmake_quote(case['name'])} PROPERTIES {key} {cmake_quote(value)})")
    return '\n'.join(lines) + '\n'


def classify_junit(path, expected, returncode):
    try:
        tests = list(ET.parse(path).getroot().iter('testcase'))
    except (OSError, ET.ParseError):
        return 'INFRA', 0.0, 'Missing or invalid CTest JUnit output'
    if len(tests) != 1 or tests[0].get('name') != expected:
        return 'INFRA', 0.0, 'JUnit does not contain exactly the requested test'
    test = tests[0]
    duration = float(test.get('time', 0))
    if test.find('skipped') is not None or test.get('status') == 'notrun':
        return 'SKIPPED', duration, 'CTest skipped the test'
    if test.find('error') is not None:
        return 'INFRA', duration, 'CTest could not execute the test'
    if test.find('failure') is not None:
        return 'FAIL', duration, (test.find('failure').get('message') or 'CTest failure')
    if returncode != 0 or test.get('status') not in (None, 'run'):
        return 'INFRA', duration, 'CTest exit status disagrees with JUnit'
    return 'PASS', duration, ''


def copy_working_directory(source, destination, ancestors=()):
    """Copy fixtures and executable bits, without filesystem-specific ACL/xattrs."""
    resolved = source.resolve()
    if resolved in ancestors:
        raise ValueError('Cycle in the test working directory')
    destination.mkdir()
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            copy_working_directory(item, target, (*ancestors, resolved))
        elif item.is_file():
            shutil.copyfile(item, target)
            target.chmod(stat.S_IMODE(item.stat().st_mode))
        else:
            raise ValueError('Unsupported working-directory entry: ' + str(item))


def execute_case(root, group, index):
    plan = json.loads((root / 'results/plan.json').read_text())
    case_id = plan['groups'][group]['cases'][index]
    case = next(c for c in plan['cases'] if c['id'] == case_id)
    out = root / 'results/cases' / case_id
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    row = {'id': case_id, 'name': case['name'], 'group': group, 'ranks': case['ranks'],
           'array_index': index, 'job_id': os.environ['SLURM_ARRAY_JOB_ID'] + '_' + str(index),
           'state': 'INFRA', 'duration': 0.0, 'message': 'Execution did not complete', 'started_at': started}
    write_json(out / 'case.json', case)
    try:
        for filename, command in [
            ('allocation.txt', ['scontrol', 'show', 'job', '-dd', row['job_id']]),
            ('gpus.txt', ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader']),
            ('rank-gpu-mapping.txt', [str(root / 'control/mpiexec.sh'), '-n', str(case['ranks']), str(root / 'build/gpu_probe')]),
        ]:
            with (out / filename).open('w') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
        mapping = (out / 'rank-gpu-mapping.txt').read_text()
        uuids = re.findall(r'\buuid=([0-9a-f]{32})\b', mapping)
        ranks = sorted(map(int, re.findall(r'\brank=(\d+)\b', mapping)))
        if len(uuids) != case['ranks'] or len(set(uuids)) != case['ranks'] or ranks != list(range(case['ranks'])):
            raise ValueError('MPI ranks did not receive distinct GPUs')
        with tempfile.TemporaryDirectory(prefix='ctest-', dir=os.environ.get('TMPDIR')) as temporary:
            private = Path(temporary)
            work = private / 'work'
            original_work = Path(properties(case)['WORKING_DIRECTORY'])
            # Do not transfer shared-filesystem ACLs to node-local temporary storage.
            # The original directory remains untouched and relative fixtures are preserved.
            copy_working_directory(original_work, work)
            (private / 'CTestTestfile.cmake').write_text(test_file(case, work))
            with (out / 'ctest.log').open('w') as stream:
                result = subprocess.run(['ctest', '--test-dir', str(private), '--output-on-failure',
                                         '--parallel', '1', '--no-tests=error', '--output-junit', str(out / 'ctest.xml')],
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=case['timeout'] + 30)
            for filename in ('LastTest.log', 'LastTestsFailed.log'):
                path = private / 'Testing/Temporary' / filename
                if path.exists():
                    shutil.copy2(path, out / filename)
            row['state'], row['duration'], row['message'] = classify_junit(out / 'ctest.xml', case['name'], result.returncode)
            row['ctest_exitcode'] = result.returncode
    except Exception as error:
        row.update(state='INFRA', message=str(error))
    row.update(finished_at=time.time(), elapsed_seconds=time.time() - started)
    write_json(out / 'result.json', row)
    print(json.dumps(row), flush=True)
    return 0 if row['state'] == 'PASS' else 1


def report(root, jobs, errors):
    out = root / 'results'
    request = json.loads((root / 'request.json').read_text())
    plan_path = out / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {'cases': [], 'groups': {}}
    rows = []
    junit = ET.Element('testsuite', name='LibDDLA GPU validation')
    for case in plan['cases']:
        path = out / 'cases' / case['id'] / 'result.json'
        job_id = jobs.get(case['group'], {}).get('job_id', '')
        task_id = f"{job_id}_{case['array_index']}" if job_id else ''
        row = {'id': case['id'], 'name': case['name'], 'group': case['group'], 'ranks': case['ranks'],
               'job_id': task_id, 'state': 'INFRA', 'duration': 0.0, 'message': 'No result was produced'}
        if path.exists():
            actual = json.loads(path.read_text())
            if actual['id'] == case['id'] and actual['name'] == case['name'] and actual['job_id'] == task_id and actual['state'] in STATES:
                row.update(actual)
        accounting = jobs.get(case['group'], {}).get('tasks', {}).get(task_id, {})
        row['slurm_state'] = accounting.get('state', 'UNKNOWN')
        if row['state'] == 'PASS' and (row['slurm_state'] != 'COMPLETED' or accounting.get('exit_code') != '0:0'):
            row.update(state='INFRA', message='Passing CTest has no successful Slurm completion')
        row['log'] = f"cases/{case['id']}/ctest.log"
        rows.append(row)
        node = ET.SubElement(junit, 'testcase', name=row['name'], classname=row['group'], time=str(row['duration']))
        if row['state'] == 'FAIL':
            ET.SubElement(node, 'failure', message=row['message'])
        elif row['state'] == 'INFRA':
            ET.SubElement(node, 'error', message=row['message'])
        elif row['state'] == 'SKIPPED':
            ET.SubElement(node, 'skipped', message=row['message'])
        log = out / row['log']
        if log.exists():
            ET.SubElement(node, 'system-out').text = log.read_text(errors='replace')
    counts = {state: sum(row['state'] == state for row in rows) for state in STATES}
    success = bool(rows) and counts['PASS'] == len(rows) and not errors
    result = {'request': request, 'success': success, 'total': len(rows), 'counts': counts,
              'registered_gpu': plan.get('registered_gpu', 0), 'extra_gpu': plan.get('extra_gpu', 0),
              'excluded': plan.get('excluded', []), 'jobs': jobs, 'errors': errors, 'cases': rows,
              'elapsed_seconds': time.time() - request['created_at']}
    write_json(out / 'result.json', result)
    junit.set('tests', str(len(rows)))
    junit.set('failures', str(counts['FAIL']))
    junit.set('errors', str(counts['INFRA']))
    junit.set('skipped', str(counts['SKIPPED']))
    ET.ElementTree(junit).write(out / 'ctest.xml', encoding='utf-8', xml_declaration=True)
    lines = ['# LibDDLA daily GPU validation', '',
             f"Source: `{request['source_repository']}:{request['source_branch']}` @ `{request['source_sha']}`",
             f"Controller: `{request['control_sha']}`; run `{request['run_id']}`.",
             'Partition: `16V100`; QoS: `flood-gpu`; CUDA 12.9.1; no explicit Slurm memory request.', '',
             f"**{counts['PASS']}/{len(rows)} PASS** · {counts['FAIL']} FAIL · {counts['INFRA']} INFRA · {counts['SKIPPED']} SKIPPED",
             f"GPU inventory: {result['registered_gpu']} registered + {result['extra_gpu']} six-rank cases; "
             f"{len(result['excluded'])} CPU/unlabelled entries excluded.",
             f"Elapsed (including queue/build): {result['elapsed_seconds'] / 60:.1f} min.", '',
             '## Build and arrays', '', '| Group | Slurm job | MPI ranks | Allocated GPUs | Cases | Concurrency |',
             '|---|---|---:|---:|---:|---:|']
    lines.append(f"| build | `{jobs.get('build', {}).get('job_id', '-')}` | 4 | 4 | — | 1 |")
    for name, group in plan['groups'].items():
        lines.append(f"| {name} | `{jobs.get(name, {}).get('job_id', '-')}` | {group['ranks']} | {group['gpus']} | {len(group['cases'])} | {group['parallelism']} |")
    if errors:
        lines += ['', '## Infrastructure errors', ''] + ['- ' + e for e in errors]
    unsuccessful = [r for r in rows if r['state'] != 'PASS']
    if unsuccessful:
        lines += ['', '## Unsuccessful cases', '', '| Case | State | Reason | Log |', '|---|---|---|---|']
        for row in unsuccessful:
            message = row['message'].replace('|', '/').replace('\n', ' ')
            lines.append(f"| `{row['name']}` | {row['state']} | {message} | `{row['log']}` |")
    lines += ['', '## All GPU cases', '', '| # | Case | Ranks | State | Seconds | Slurm task | Log |',
              '|---:|---|---:|---|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['id']} | `{row['name']}` | {row['ranks']} | {row['state']} | {row['duration']:.2f} | `{row['job_id']}` | `{row['log']}` |")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan', 'case'))
    parser.add_argument('root', type=Path)
    parser.add_argument('group', nargs='?')
    parser.add_argument('index', type=int, nargs='?')
    args = parser.parse_args()
    if args.action == 'plan':
        inventory = json.loads((args.root / 'results/inventory.json').read_text())
        write_json(args.root / 'results/plan.json', make_plan(inventory, config(args.root / 'control/config.ini'), args.root))
    else:
        raise SystemExit(execute_case(args.root, args.group, args.index))
