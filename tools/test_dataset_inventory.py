import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('dataset_inventory.py')


class InventoryTests(unittest.TestCase):
    def test_resource_limited_json_does_not_abort_scan(self):
        fixtures = {
            'oversized_integer': '9' * 5000,
            'deep_nesting': '[' * 5000 + '0' + ']' * 5000,
        }
        for name, contents in fixtures.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / 'data'
                data.mkdir()
                problem = data / 'a_problem.json'
                problem.write_text(contents, encoding='utf-8')
                valid = {'confidence': 1, 'sequenceData': [[0] * 17] * 15}
                (data / 'z_valid.json').write_text(json.dumps(valid), encoding='utf-8')
                report = root / 'report.json'
                result = subprocess.run(
                    [sys.executable, '-X', 'int_max_str_digits=4300',
                     str(SCRIPT), str(data), '--report', str(report)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(report.exists())
                output = json.loads(report.read_text(encoding='utf-8'))
                self.assertEqual(output['summary']['file_count'], 2)
                self.assertEqual(output['summary']['recording_count'], 1)
                self.assertEqual(output['summary']['error_count'], 1)
                self.assertEqual(output['invalid_files'], ['a_problem.json'])
                self.assertEqual(output['groups']['.']['confidence_label_counts']['1'], 1)
                self.assertEqual(problem.read_text(encoding='utf-8'), contents)

    def test_scan_continues_and_reports_locations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            nested = data / 'nested'
            nested.mkdir(parents=True)
            (data / 'bad.json').write_text('{\n broken', encoding='utf-8')
            (data / 'empty.json').write_text('[]', encoding='utf-8')
            (data / 'object.json').write_text('{}', encoding='utf-8')
            frame = [0] * 17
            frame[5] = -1
            frame[3] = -1
            recording = {'confidence': 1, 'sequenceData': [frame] * 15}
            (nested / 'valid.json').write_text(json.dumps(recording, indent=2), encoding='utf-8')
            (nested / 'invalid.json').write_text('{\n "confidence": 2,\n "sequenceData": [[true]]\n}', encoding='utf-8')
            report = root / 'report.json'
            result = subprocess.run([sys.executable, str(SCRIPT), str(data), '--report', str(report)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(output['summary']['file_count'], 5)
            self.assertEqual(output['groups']['nested']['recording_count'], 2)
            self.assertTrue(any(i['file'] == 'bad.json' and i['line'] == 2 for i in output['issues']))
            self.assertTrue(any(i['file'] == 'nested/invalid.json' and i.get('feature_index') == 0 and i['line'] == 3 for i in output['issues']))
            self.assertTrue(any('Negative opposition' in i['message'] for i in output['issues']))
            self.assertTrue(any(i.get('feature_index') == 5 and 'Negative opposition' in i['message'] for i in output['issues']))
            self.assertFalse(any(i.get('feature_index') == 3 and 'Negative opposition' in i['message'] for i in output['issues']))
            self.assertEqual([i['severity'] for i in output['issues'] if i['file'] == 'object.json'], ['warning'])
            self.assertIn('additional issues found', result.stdout)

    def test_report_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.json'
            source.write_text('[]', encoding='utf-8')
            result = subprocess.run([sys.executable, str(SCRIPT), str(root), '--report', str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_text(encoding='utf-8'), '[]')

    def test_existing_report_outside_scan_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            data.mkdir()
            report = root / 'existing-report.json'
            original = '{"keep": "existing report"}'
            report.write_text(original, encoding='utf-8')
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(data), '--report', str(report)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Report already exists', result.stderr)
            self.assertEqual(report.read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
