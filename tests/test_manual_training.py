"""Import real manual-format PNGs without inventing contact metadata or groups."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.camera import FramePacket
from core.inference import CLASS_NAMES, RAW_INPUT_KIND
from core.manual_collection import ManualDatasetWriter
from training.data import prepare_dataset, read_image, read_samples, _reference_split
from training.calibrate import evaluate_top1


class ManualTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'manual'
        self.output = Path(self.temp.name) / 'prepared'

    def write_session(self, labels=CLASS_NAMES, offset=0):
        writer = ManualDatasetWriter(self.root)
        for index, label in enumerate(labels):
            for n in range(5):
                frame = np.full((24, 32, 3), offset + index*5 + n, dtype=np.uint8)
                frame[2:7, 5:9] = 255
                writer.save(FramePacket(1, index*5+n, 1., frame), label, '2026-11-25T10:00:00+08:00')
        return writer

    def test_raw_import_keeps_orientation_pixels_and_manual_provenance(self):
        writer = self.write_session()
        samples = read_samples(self.root, input_kind=RAW_INPUT_KIND, manual_grouping='frame')
        rows = [json.loads(line) for line in (writer.session_dir / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(samples), len(rows))
        for sample, row in zip(samples, rows):
            np.testing.assert_array_equal(read_image(sample['path']), read_image(writer.session_dir / row['image']))
            self.assertEqual(sample['label_source'], 'manual')
            self.assertNotIn('contact_detected', sample)
            self.assertNotIn('baseline', sample)
        self.assertEqual(len({s['episode_id'] for s in samples if s['label'] == 'no_contact'}), 1)
        self.assertEqual(len({s['episode_id'] for s in samples if s['label'] == 'rat'}), 5)

    def test_manual_images_cannot_be_used_as_differences(self):
        self.write_session()
        with self.assertRaisesRegex(ValueError, '--input raw'):
            read_samples(self.root)

    def test_default_grouping_never_assumes_frames_are_independent_presses(self):
        self.write_session()
        with self.assertRaisesRegex(ValueError, '无法按完整'):
            prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND)

    def test_independent_press_split_keeps_empty_frames_and_class_indices(self):
        self.write_session()
        split, report = prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND, manual_grouping='frame')
        self.assertEqual(report['grouping'], 'episode')
        self.assertEqual(report['manual_grouping'], 'frame')
        self.assertEqual(report['missing_validation_classes'], ['no_contact'])
        self.assertEqual(sum(row['label'] == 'no_contact' for row in split['train']), 5)
        groups = [{(s['session_id'], s['episode_id']) for s in split[name]} for name in ('train', 'val')]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        for name in ('train', 'val'):
            self.assertEqual(sorted(p.name for p in (self.output / name).iterdir() if p.is_dir()), list(CLASS_NAMES))
            provenance = [json.loads(line) for line in (self.output / name / 'index.jsonl').read_text().splitlines()]
            self.assertEqual(len(provenance), len(split[name]))
            self.assertTrue(all((self.output / row['prepared_path']).is_file() for row in provenance))

    def test_missing_unknown_requires_explicit_opt_in_and_is_reported(self):
        self.write_session([label for label in CLASS_NAMES if label != 'unknown'])
        with self.assertRaisesRegex(ValueError, 'unknown'):
            prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND, manual_grouping='frame')
        _, report = prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND,
                                    manual_grouping='frame', allow_missing_unknown=True)
        self.assertEqual(report['missing_training_classes'], ['unknown'])
        self.assertEqual(report['missing_validation_classes'], ['no_contact', 'unknown'])
        self.assertTrue((self.output / 'train' / 'unknown').is_dir())

    def test_missing_unknown_option_does_not_allow_missing_zodiac(self):
        self.write_session([label for label in CLASS_NAMES if label not in ('rat', 'unknown')])
        with self.assertRaisesRegex(ValueError, 'rat'):
            prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND,
                            manual_grouping='frame', allow_missing_unknown=True)

    def test_two_sessions_prefer_session_split_even_with_frame_option(self):
        self.write_session(offset=0)
        self.write_session(offset=100)
        split, report = prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND, manual_grouping='frame')
        self.assertEqual(report['grouping'], 'session')
        self.assertTrue({s['session_id'] for s in split['train']}.isdisjoint({s['session_id'] for s in split['val']}))

    def test_incremental_split_keeps_old_validation_and_groups_new_presses(self):
        self.write_session()
        previous, _ = prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND,
                                      manual_grouping='frame')
        self.write_session(offset=100)
        current, report = prepare_dataset(self.root, self.output.with_name('incremental'),
            input_kind=RAW_INPUT_KIND, manual_grouping='frame', reference_split=self.output)
        for name in ('train', 'val'):
            old_hashes = {r['sha256'] for r in previous[name]}
            self.assertTrue(old_hashes <= {r['sha256'] for r in current[name]})
            self.assertGreater(report['reference_split']['added'][name], 0)
        self.assertEqual(report['grouping'], 'reference+episode')
        self.assertTrue({r['sha256'] for r in current['train']}.isdisjoint(
            {r['sha256'] for r in current['val']}))
        self.assertTrue({(r['session_id'], r['episode_id']) for r in current['train']}.isdisjoint(
            {(r['session_id'], r['episode_id']) for r in current['val']}))

    def test_reference_split_rejects_missing_relabelled_or_crossed_episode(self):
        self.write_session()
        previous, _ = prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND,
                                      manual_grouping='frame')
        rows = previous['train'] + previous['val']
        with self.assertRaisesRegex(ValueError, '图像已缺失'):
            _reference_split(rows[1:], self.output, seed=42, min_per_class=1, input_kind=RAW_INPUT_KIND)
        changed = [{**r, 'label': 'tiger'} if i == 0 else r for i, r in enumerate(rows)]
        with self.assertRaisesRegex(ValueError, '标签已改变'):
            _reference_split(changed, self.output, seed=42, min_per_class=1, input_kind=RAW_INPUT_KIND)
        changed = [{**r, 'episode_id': 'one-press'} for r in rows]
        with self.assertRaisesRegex(ValueError, '同一次按压'):
            _reference_split(changed, self.output, seed=42, min_per_class=1, input_kind=RAW_INPUT_KIND)

    def test_new_frame_in_old_validation_episode_never_enters_training(self):
        self.write_session()
        previous, _ = prepare_dataset(self.root, self.output, input_kind=RAW_INPUT_KIND,
                                      manual_grouping='frame')
        new = {**previous['val'][0], 'sha256': 'new-pixels-in-held-out-press'}
        split, details = _reference_split(previous['train']+previous['val']+[new], self.output,
                                         seed=42, min_per_class=1, input_kind=RAW_INPUT_KIND)
        self.assertIn(new, split['val'])
        self.assertNotIn(new, split['train'])
        self.assertEqual(details['added'], {'train': 0, 'val': 1})

    def test_wrong_orientation_and_path_escape_rejected(self):
        writer = self.write_session()
        file = writer.session_dir / 'samples.jsonl'
        original = file.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        for field, value in [('rotation_degrees', 0), ('image', '../../outside.png'), ('frame_shape', [1, 1, 3])]:
            bad = {**rows[0], field: value}
            file.write_text(json.dumps(bad) + '\n')
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_samples(self.root, input_kind=RAW_INPUT_KIND)
        file.write_text(original)

    def test_top1_reports_wrong_predictions_and_missing_negatives(self):
        probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        probabilities[CLASS_NAMES.index('rat')] = 1.
        report = evaluate_top1([{'label': 'rat', 'probabilities': probabilities},
                                {'label': 'ox', 'probabilities': probabilities}])
        self.assertEqual(report['accuracy'], .5)
        self.assertEqual(report['confusion']['ox'], {'rat': 1})
        self.assertIn('unknown', report['missing_classes'])
        self.assertIsNone(evaluate_top1([])['accuracy'])


if __name__ == '__main__':
    unittest.main()
