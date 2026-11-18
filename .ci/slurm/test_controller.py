import configparser
import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from coordinator import cancel, done, job_states, parse_accounting
from publish import admission
from runtime import classify_junit, copy_working_directory, make_plan, report, test_file, write_json
from wait import collect_results


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = configparser.ConfigParser()
        self.cfg.read(Path(__file__).with_name('config.ini'))
        for section in self.cfg.sections():
            if section.startswith('extra.'):
                path = self.root / 'build/tests' / self.cfg[section]['binary']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    def case(self, name='gpu_four', ranks=4, labels=('gpu',)):
        return {'name': name, 'command': [str(self.root / 'control/mpiexec.sh'), '-n', str(ranks), '/binary'],
                'properties': [{'name': 'LABELS', 'value': list(labels)},
                               {'name': 'PROCESSORS', 'value': ranks * 4},
                               {'name': 'WORKING_DIRECTORY', 'value': '/build/tests'},
                               {'name': 'ENVIRONMENT', 'value': ['LD_PRELOAD=/lib/trace.so', 'OMP_NUM_THREADS=1']},
                               {'name': 'SKIP_RETURN_CODE', 'value': 77}]}

    def test_gpu_only_and_actual_mpi_ranks(self):
        tests = [self.case(), self.case('gpu_one', 1), self.case('gpu_two', 2),
                 self.case('cpu', 4, ('cpu',)), self.case('unlabelled', 4, ())]
        plan = make_plan({'tests': tests}, self.cfg, self.root)
        self.assertEqual((plan['registered_gpu'], plan['extra_gpu']), (3, 2))
        self.assertEqual(plan['excluded'], ['cpu', 'unlabelled'])
        self.assertEqual({k: len(v['cases']) for k, v in plan['groups'].items()},
                         {'mpi1': 1, 'mpi2': 1, 'mpi4': 1, 'mpi6': 2})
        self.assertEqual([c['ranks'] for c in plan['cases']], [4, 1, 2, 6, 6])
        self.assertEqual(plan['cases'][0]['command'], tests[0]['command'])

    def test_empty_gpu_inventory_fails(self):
        with self.assertRaisesRegex(ValueError, 'no GPU'):
            make_plan({'tests': [self.case(labels=('cpu',))]}, self.cfg, self.root)

    def test_new_unconfigured_resources_and_fixtures_fail_visibly(self):
        with self.assertRaisesRegex(ValueError, 'No resource'):
            make_plan({'tests': [self.case(ranks=8)]}, self.cfg, self.root)
        case = self.case()
        case['properties'].append({'name': 'FIXTURES_REQUIRED', 'value': ['database']})
        with self.assertRaisesRegex(ValueError, 'Unsupported CTest'):
            make_plan({'tests': [case]}, self.cfg, self.root)

    def test_duplicate_tests_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            make_plan({'tests': [self.case(), self.case()]}, self.cfg, self.root)

    def test_isolated_ctest_preserves_failure_and_environment_properties(self):
        plan = make_plan({'tests': [self.case()]}, self.cfg, self.root)
        text = test_file(plan['cases'][0], Path('/private/work'))
        self.assertIn('LD_PRELOAD=/lib/trace.so;OMP_NUM_THREADS=1', text)
        self.assertIn('SKIP_RETURN_CODE [=[77]=]', text)
        self.assertIn('WORKING_DIRECTORY [=[/private/work]=]', text)

    def classify(self, body, rc=0):
        path = self.root / 'ctest.xml'
        path.write_text(body)
        return classify_junit(path, 'expected', rc)[0]

    def test_junit_pass_failure_skip_and_mismatch(self):
        self.assertEqual(self.classify('<testsuite><testcase name="expected" status="run" time="1"/></testsuite>'), 'PASS')
        self.assertEqual(self.classify('<testsuite><testcase name="expected"><failure/></testcase></testsuite>', 8), 'FAIL')
        self.assertEqual(self.classify('<testsuite><testcase name="expected"><skipped/></testcase></testsuite>'), 'SKIPPED')
        self.assertEqual(self.classify('<testsuite><testcase name="other"/></testsuite>'), 'INFRA')
        self.assertEqual(self.classify('<testsuite/>'), 'INFRA')
        self.assertEqual(self.classify('<invalid'), 'INFRA')
        self.assertEqual(self.classify('<testsuite><testcase name="expected"/></testsuite>', 1), 'INFRA')

    def test_accounting_uses_array_task_ids_and_checks_job_identity(self):
        tasks = parse_accounting('123_0|our-array|COMPLETED|0:0\n123_1|our-array|FAILED|1:0\n', {'123': 'our-array'})
        self.assertTrue(done({'job_id': '123', 'array_size': 2, 'tasks': tasks}))
        self.assertFalse(done({'job_id': '123', 'array_size': 3, 'tasks': tasks}))
        with self.assertRaisesRegex(ValueError, 'identity'):
            parse_accounting('123_0|other-job|RUNNING|0:0', {'123': 'our-array'})

    def test_finished_job_no_longer_in_squeue_uses_accounting(self):
        jobs = {'build': {'job_id': '123', 'name': 'build', 'array_size': 0}}
        with patch('coordinator.capture', return_value='123|build|COMPLETED|0:0'), \
                patch('coordinator.subprocess.run') as queue:
            job_states(jobs)
            self.assertTrue(done(jobs['build']))
            queue.assert_not_called()

    def test_accounting_lag_after_queue_removal_is_not_a_failure(self):
        jobs = {'build': {'job_id': '123', 'name': 'build', 'array_size': 0}}
        with patch('coordinator.capture', return_value='123|build|RUNNING|0:0'), \
                patch('coordinator.subprocess.run') as queue:
            queue.return_value.returncode = 1
            queue.return_value.stdout = ''
            queue.return_value.stderr = 'slurm_load_jobs error: Invalid job id specified'
            job_states(jobs)
            self.assertFalse(done(jobs['build']))

    def test_completion_race_does_not_prevent_cancelling_other_arrays(self):
        jobs = {name: {'job_id': number, 'name': name, 'array_size': 2, 'tasks': {}}
                for name, number in [('already_finished', '123'), ('still_active', '456')]}
        with patch('coordinator.subprocess.run', side_effect=[
                subprocess.CompletedProcess([], 1, '', 'Invalid job id specified'),
                subprocess.CompletedProcess([], 0, '', '')]) as command:
            cancel(jobs)
            self.assertEqual(command.call_count, 2)
            self.assertEqual(command.call_args.args[0], ['scancel', '--name=still_active', '456'])

    def test_missing_result_and_slurm_failure_cannot_pass(self):
        plan = make_plan({'tests': [self.case('missing'), self.case('bad_slurm')]}, self.cfg, self.root)
        write_json(self.root / 'results/plan.json', plan)
        write_json(self.root / 'request.json', {'source_repository': 'upstream/repo', 'source_branch': 'develop',
                                              'source_sha': 'a' * 40, 'control_sha': 'b' * 40, 'run_id': 'gh-1-1', 'created_at': 0})
        case = plan['cases'][1]
        write_json(self.root / 'results/cases' / case['id'] / 'result.json',
                   {'id': case['id'], 'name': case['name'], 'job_id': '123_1', 'state': 'PASS', 'duration': 1})
        result = report(self.root, {'mpi4': {'job_id': '123', 'tasks': {'123_1': {'state': 'FAILED', 'exit_code': '1:0'}}}}, [])
        self.assertFalse(result['success'])
        self.assertEqual(result['counts']['PASS'], 0)
        self.assertEqual(result['counts']['INFRA'], 4)

    def test_only_installed_fork_main_schedule_or_manual_is_admitted(self):
        run = {'repository': {'full_name': 'Stardust0831/LibDDLA'},
               'head_repository': {'full_name': 'Stardust0831/LibDDLA'},
               'path': '.github/workflows/gpu-validation.yml', 'event': 'schedule',
               'head_branch': 'main', 'head_sha': 'a' * 40}
        admission(run, 'a' * 40)
        for field, value in [('event', 'pull_request'), ('head_branch', 'develop'), ('head_sha', 'b' * 40)]:
            changed = copy.deepcopy(run)
            changed[field] = value
            with self.assertRaises(ValueError):
                admission(changed, 'a' * 40)

    def test_cancelled_run_still_collects_available_progress(self):
        result, target = self.root / 'public', self.root / 'artifact'
        result.mkdir()
        (result / 'progress.log').write_text('build running\n')
        collect_results(result, target, self.root / 'step-summary.md')
        self.assertEqual((target / 'progress.log').read_text(), 'build running\n')
        self.assertIn('INCOMPLETE', (target / 'remote/summary.md').read_text())

    def test_completed_report_is_published_without_replacing_failure(self):
        result, target = self.root / 'public', self.root / 'artifact'
        (result / 'remote').mkdir(parents=True)
        (result / 'remote/summary.md').write_text('177 PASS / 2 FAIL\n')
        collect_results(result, target, self.root / 'step-summary.md')
        self.assertEqual((self.root / 'step-summary.md').read_text(), '177 PASS / 2 FAIL\n')

    def test_working_copy_preserves_fixtures_and_execute_bits_without_acls(self):
        source, target = self.root / 'original', self.root / 'private'
        (source / 'data').mkdir(parents=True)
        (source / 'data/input').write_text('matrix fixture')
        (source / 'helper').write_text('#!/bin/sh\nexit 0\n')
        (source / 'helper').chmod(0o755)
        with patch('shutil.copystat', side_effect=AssertionError('ACL metadata must not be copied')):
            copy_working_directory(source, target)
        self.assertEqual((target / 'data/input').read_text(), 'matrix fixture')
        self.assertEqual((target / 'helper').stat().st_mode & 0o777, 0o755)
        (target / 'data/input').write_text('changed by test')
        self.assertEqual((source / 'data/input').read_text(), 'matrix fixture')


if __name__ == '__main__':
    unittest.main()
