import errno
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.inference import RAW_INPUT_KIND, file_sha256
from core.portable_model import activate, rollback, selected
from training.data import prepare_dataset, read_samples
from training.portable import load_snapshot, merge_new, pixel_hash, save_snapshot
from train_easy import check_retention


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def row(self, n, label='rat', session='old', episode=None):
        frame = np.full((20, 24, 3), n, np.uint8)
        path = self.root / f'{n}.png'
        cv2.imencode('.png', frame)[1].tofile(path)
        return dict(path=path, label=label, session_id=session, episode_id=episode or f'press{n}',
                    sha256=pixel_hash(frame))

    def test_snapshot_relocates_preserves_groups_and_checks_pixels(self):
        rows = [self.row(1), self.row(2)]
        source = self.root / 'source'
        save_snapshot(rows, source)
        target = self.root / '中文 新位置'
        shutil.move(source, target)
        loaded = read_samples(target, input_kind=RAW_INPUT_KIND)
        self.assertEqual([r['sha256'] for r in loaded], [r['sha256'] for r in rows])
        self.assertEqual([r['episode_id'] for r in loaded], ['press1', 'press2'])
        shutil.copy2(loaded[0]['path'], loaded[1]['path'])
        with self.assertRaisesRegex(ValueError, '修改'):
            load_snapshot(target)

    def test_snapshot_rejects_escape_paths(self):
        target = self.root / 'source'
        save_snapshot([self.row(1)], target)
        path = target / 'snapshot.json'
        data = json.loads(path.read_text())
        data['samples'][0]['image'] = '../1.png'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, '越界'):
            load_snapshot(target)

    def test_merging_keeps_known_holdout_group_and_rejects_conflicts(self):
        old = self.row(1)
        changed_group = {**old, 'episode_id': 'manual-session'}
        merged, added = merge_new([old], [changed_group, self.row(2, session='new')])
        self.assertEqual(merged[0]['episode_id'], 'press1')
        self.assertEqual(len(added), 1)
        with self.assertRaisesRegex(ValueError, '标签冲突'):
            merge_new([old], [{**old, 'label': 'ox'}])
        external = self.row(3)
        with self.assertRaisesRegex(ValueError, '独立回归'):
            merge_new([old], [external], [external])
        with self.assertRaisesRegex(ValueError, '独立回归'):
            merge_new([old], [self.row(4, episode='press3')], [external])

    def test_reference_partition_survives_move_and_new_session_stays_together(self):
        rows = [self.row(1), self.row(2), self.row(3, session='new', episode='manual-session'),
                self.row(4, session='new', episode='manual-session')]
        source = self.root / 'source'
        save_snapshot(rows, source)
        reference = self.root / 'reference'
        reference.mkdir()
        (reference / 'split.json').write_text(json.dumps({'input_kind': RAW_INPUT_KIND}))
        for part, row in zip(('train', 'val'), rows[:2]):
            (reference / part).mkdir()
            (reference / part / 'index.jsonl').write_text(json.dumps(
                {'sha256': row['sha256'], 'label': row['label']}))
        split, report = prepare_dataset(source, self.root / 'out', reference_split=reference,
                                         input_kind=RAW_INPUT_KIND, allow_missing_unknown=True)
        self.assertEqual([r['sha256'] for r in split['val']], [rows[1]['sha256']])
        self.assertEqual(len(split['train']), 3)
        self.assertEqual(report['reference_split']['retained'], {'train': 1, 'val': 1})

    def test_retention_checks_each_image_and_confidence_not_only_mean(self):
        row = self.row(1)
        key = row['sha256']
        before = {key: {'label': 'rat', 'confidence': .9}}
        for label, confidence in (('ox', .99), ('rat', .49), ('rat', .74)):
            self.assertTrue(check_retention([row], before, {key: dict(label=label, confidence=confidence)}))
        self.assertEqual(check_retention([row], before, {key: dict(label='rat', confidence=.8)}), [])

    def make_model(self, name):
        directory = self.root / name
        directory.mkdir(parents=True)
        (directory / 'best.onnx').write_bytes(name.encode())
        (directory / 'best.json').write_text(json.dumps({'sha256': file_sha256(directory / 'best.onnx')}))
        (directory / 'best.pt').write_bytes(b'weights')
        (self.root / 'data/seed/reference').mkdir(parents=True, exist_ok=True)
        return dict(model_dir=name, snapshot='data/seed', reference='data/seed/reference',
                    weights_sha256=file_sha256(directory / 'best.pt'))

    def test_activation_is_atomic_and_rollback_keeps_matching_weights_and_data(self):
        original = self.make_model('models/raw')
        candidate = self.make_model('runs/example')
        activate(self.root, candidate, original)
        self.assertEqual(selected(self.root)['model_dir'], candidate['model_dir'])
        self.assertTrue(rollback(self.root))
        self.assertEqual(selected(self.root)['model_dir'], original['model_dir'])
        pointer = (self.root / 'models/active.json').read_bytes()
        (self.root / 'runs/example/best.pt').write_bytes(b'wrong weights')
        with self.assertRaisesRegex(ValueError, '校验失败'):
            activate(self.root, candidate, original)
        self.assertEqual((self.root / 'models/active.json').read_bytes(), pointer)

    def test_windows_lock_contention_reads_metadata_without_releasing_other_lock(self):
        from core.instance import desktop_instance
        with patch.object(sys, 'platform', 'win32'), patch('core.instance._lock') as lock:
            lock.side_effect = [None, BlockingIOError(errno.EAGAIN, 'locked'), None]
            with desktop_instance(self.root / 'lock', 'url') as owner:
                self.assertIsNone(owner)
                with desktop_instance(self.root / 'lock', 'second') as duplicate:
                    self.assertEqual(duplicate['url'], 'url')
        self.assertEqual(lock.call_count, 3)
        self.assertEqual(lock.call_args.kwargs, {'unlock': True})


if __name__ == '__main__':
    unittest.main()
