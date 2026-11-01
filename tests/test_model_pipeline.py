"""Behavioral tests for no-forced-classification and data isolation guarantees."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.inference import (CLASS_NAMES, INPUT_KIND, PREPROCESS_VERSION, RAW_INPUT_KIND,
                            OnnxClassifier, decide, file_sha256, preprocess)
from core.zodiac import ZODIAC_KEYS
from training.calibrate import calibrate, evaluate, passes_release
from training.data import prepare_dataset, read_image, read_samples, split_sessions


def temporary_directory():
    root = Path(__file__).resolve().parents[1] / '.cache' / 'model-tests'
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def fixture_manifest():
    return {'schema_version': 1, 'status': 'validated', 'synthetic': False,
            'class_names': list(CLASS_NAMES), 'preprocess': PREPROCESS_VERSION,
            'input_kind': INPUT_KIND, 'image_size': 32,
            'prototypes': {key: [1.0, 0.0] for key in ZODIAC_KEYS},
            'thresholds': {key: {'confidence': 0.85, 'margin': 0.2, 'distance': 0.1} for key in ZODIAC_KEYS}}


def distribution(key, confidence=0.95):
    result = np.full(14, (1 - confidence) / 13, dtype=np.float32)
    result[CLASS_NAMES.index(key)] = confidence
    return result


class RejectionTests(unittest.TestCase):
    def test_unknown_and_empty_are_never_zodiacs(self):
        for key in ('unknown', 'no_contact'):
            result = decide(distribution(key, 0.99), np.array([1, 0]), fixture_manifest())
            self.assertFalse(result.accepted)
            self.assertIsNone(result.key)
            self.assertEqual(result.reason, key)

    def test_high_confidence_unfamiliar_embedding_is_rejected(self):
        result = decide(distribution('rat', 0.999), np.array([0, 1]), fixture_manifest())
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'out_of_distribution')

    def test_low_confidence_and_small_margin_are_rejected(self):
        self.assertEqual(decide(distribution('rat', 0.7), np.array([1, 0]), fixture_manifest()).reason, 'low_confidence')
        manifest = fixture_manifest()
        manifest['thresholds']['rat']['confidence'] = 0.4
        p = np.zeros(14, dtype=np.float32)
        p[CLASS_NAMES.index('rat')], p[CLASS_NAMES.index('ox')] = 0.55, 0.45
        self.assertEqual(decide(p, np.array([1, 0]), manifest).reason, 'ambiguous')

    def test_valid_known_pattern_passes(self):
        result = decide(distribution('rat'), np.array([1, 0]), fixture_manifest())
        self.assertTrue(result.accepted)
        self.assertEqual(result.key, 'rat')

    def test_preview_shows_top1_including_negative_classes(self):
        for label in ('rat', 'unknown', 'no_contact'):
            with self.subTest(label=label):
                result = decide(distribution(label, 0.4), np.array([0, 1]), fixture_manifest(), preview=True)
                self.assertEqual(result.predicted_key, label)
                self.assertEqual(result.key, 'rat' if label == 'rat' else None)
                self.assertEqual(result.reason, 'preview_top1' if label == 'rat' else label)
                self.assertEqual(result.accepted, label == 'rat')

    def test_nonfinite_or_unormalized_output_fails_closed(self):
        for p, e in [(np.full(14, np.nan), [1, 0]), (np.ones(14), [1, 0]), (distribution('rat'), [0, 0])]:
            with self.assertRaises(ValueError):
                decide(p, np.array(e), fixture_manifest())

    def test_letterbox_preserves_edge_and_bgr_to_rgb(self):
        frame = np.zeros((16, 32, 3), np.uint8)
        frame[:, 0] = (2, 5, 255)
        tensor = preprocess(frame, 32)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertEqual(float(tensor[0, 0, 8, 0]), 1.0)
        self.assertAlmostEqual(float(tensor[0, 2, 8, 0]), 2 / 255)
        self.assertTrue(np.allclose(tensor[0, :, 0, 0], 114 / 255))

    def test_calibration_uses_validation_unknown_negatives(self):
        train = [{'label': key, 'probabilities': distribution(key), 'embedding': np.array([1, 0])} for key in ZODIAC_KEYS]
        val = list(train) + [{'label': 'unknown', 'probabilities': distribution('rat', 0.95), 'embedding': np.array([1, 0])},
                             {'label': 'no_contact', 'probabilities': distribution('no_contact'), 'embedding': np.array([1, 0])}]
        manifest, report = calibrate(train, val, fixture_manifest())
        self.assertEqual(report['false_accept_rate'], 0)
        self.assertFalse(passes_release(report))  # must not release a reject-everything model

    def test_release_requires_validation_examples_of_every_class(self):
        samples = [{'label': key, 'probabilities': distribution(key), 'embedding': np.array([1, 0])}
                   for key in CLASS_NAMES]
        manifest = fixture_manifest()
        complete = evaluate(samples, manifest)
        self.assertEqual(complete['missing_classes'], [])
        self.assertTrue(passes_release(complete))
        for missing in CLASS_NAMES:
            with self.subTest(missing=missing):
                report = evaluate([row for row in samples if row['label'] != missing], manifest)
                self.assertEqual(report['missing_classes'], [missing])
                self.assertFalse(passes_release(report, min_recall=0, max_far=1))


class DataSplitTests(unittest.TestCase):
    def write_sample(self, root, session, *, label='no_contact', contact=False, background=30):
        directory = root / session
        directory.mkdir()
        baseline = np.full((32, 32, 3), background, np.uint8)
        raw = baseline.copy()
        raw[10:15, 10:15] += 1
        for name, frame in [('image', cv2.absdiff(raw, baseline)), ('raw', raw), ('baseline', baseline)]:
            cv2.imencode('.png', frame)[1].tofile(directory / f'{name}.png')
        row = {'image': f'{session}/image.png', 'raw_image': f'{session}/raw.png', 'baseline': f'{session}/baseline.png',
               'session_id': session, 'episode_id': 'e1', 'label': label,
               'contact_detected': contact, 'input_kind': INPUT_KIND}
        (directory / 'manifest.jsonl').write_text(json.dumps(row), encoding='utf-8')

    def test_valid_empty_difference_is_read_but_insufficient_to_train(self):
        with temporary_directory() as tmp:
            self.write_sample(Path(tmp), 'empty_session')
            samples = read_samples(tmp)
            self.assertEqual(samples[0]['label'], 'no_contact')
            self.assertFalse(samples[0].get('synthetic', False))
            with self.assertRaises(ValueError):
                split_sessions(samples)

    def test_copied_frames_across_sessions_rejected(self):
        with temporary_directory() as tmp:
            self.write_sample(Path(tmp), 'session_a')
            self.write_sample(Path(tmp), 'session_b')
            with self.assertRaisesRegex(ValueError, '数据泄漏'):
                read_samples(tmp)

    def test_prepare_raw_materializes_originals_without_reading_difference_or_baseline(self):
        with temporary_directory() as tmp:
            root = Path(tmp) / 'raw'
            root.mkdir()
            for batch in range(2):
                for i, label in enumerate(CLASS_NAMES):
                    session = f'{batch}-{label}'
                    self.write_sample(root, session, label=label, contact=label != 'no_contact',
                                      background=10 + batch * len(CLASS_NAMES) + i)
                    (root / session / 'image.png').unlink()
                    (root / session / 'baseline.png').unlink()
            prepared = Path(tmp) / 'prepared'
            split, report = prepare_dataset(root, prepared, input_kind=RAW_INPUT_KIND)
            self.assertEqual(report['input_kind'], RAW_INPUT_KIND)
            self.assertEqual(json.loads((prepared / 'split.json').read_text())['input_kind'], RAW_INPUT_KIND)
            self.assertEqual(sum(map(len, split.values())), 2 * len(CLASS_NAMES))
            for part in split.values():
                for row in part:
                    np.testing.assert_array_equal(read_image(row['path']), read_image(root / row['raw_image']))

    def test_raw_duplicate_pixels_across_sessions_even_when_differences_differ(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            self.write_sample(root, 'a')
            self.write_sample(root, 'b')
            # Identical raw frame, distinct references and lossless PNG encodings.
            frame = read_image(root / 'b' / 'raw.png')
            cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tofile(root / 'b' / 'raw.png')
            cv2.imencode('.png', np.zeros_like(frame))[1].tofile(root / 'b' / 'baseline.png')
            cv2.imencode('.png', frame)[1].tofile(root / 'b' / 'image.png')
            self.assertEqual(len(read_samples(root)), 2)
            with self.assertRaisesRegex(ValueError, '数据泄漏'):
                read_samples(root, input_kind=RAW_INPUT_KIND)

    def test_reencoded_identical_pixels_do_not_bypass_leak_check(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            self.write_sample(root, 'session_a')
            self.write_sample(root, 'session_b')
            path_a, path_b = root / 'session_a/image.png', root / 'session_b/image.png'
            image = cv2.imdecode(np.fromfile(path_b, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tofile(path_b)
            self.assertNotEqual(path_a.read_bytes(), path_b.read_bytes())
            with self.assertRaisesRegex(ValueError, '数据泄漏'):
                read_samples(root)

    def test_positive_label_without_contact_rejected(self):
        with temporary_directory() as tmp:
            self.write_sample(Path(tmp), 'bad_label', label='rat', contact=False)
            with self.assertRaisesRegex(ValueError, '确有接触'):
                read_samples(tmp)

    def test_one_class_per_session_and_mixed_sessions_do_not_leak(self):
        for mixed in (True, False):
            rows = []
            for label in CLASS_NAMES:
                for session in range(5):
                    rows.extend({'label': label, 'session_id': str(session) if mixed else f'{label}_{session}'} for _ in range(10))
            split = split_sessions(rows)
            self.assertEqual(set(split), {'train', 'val'})
            self.assertEqual(len(split['train']), len(rows) * 4 // 5)
            sets = [{row['session_id'] for row in split[name]} for name in ('train', 'val')]
            self.assertTrue(sets[0].isdisjoint(sets[1]))
            self.assertTrue(all({row['label'] for row in part} == set(CLASS_NAMES) for part in split.values()))

    def test_single_calibration_keeps_press_groups_and_singleton_empty_together(self):
        rows = [{'label': label, 'session_id': 'batch', 'episode_id': f'{label}-{i // 2}'}
                for label in CLASS_NAMES if label != 'no_contact' for i in range(20)]
        rows += [{'label': 'no_contact', 'session_id': 'batch', 'episode_id': 'empty'} for _ in range(20)]
        split = split_sessions(rows)
        self.assertEqual(len(split['val']), len(rows) // 5)
        self.assertEqual({s['label'] for s in split['train']}, set(CLASS_NAMES))
        self.assertEqual({s['label'] for s in split['val']}, set(CLASS_NAMES) - {'no_contact'})
        self.assertTrue({s['episode_id'] for s in split['train']}.isdisjoint(s['episode_id'] for s in split['val']))
        # A later empty recording in its own session can cover validation.
        # Reused episode names from two different sessions are independent groups.
        rows += [{'label': 'no_contact', 'session_id': 'next', 'episode_id': 'empty'} for _ in range(4)]
        supplemented = split_sessions(rows)
        for part in supplemented.values():
            self.assertEqual({s['label'] for s in part}, set(CLASS_NAMES))
        episodes = [{(s['session_id'], s['episode_id']) for s in part} for part in supplemented.values()]
        self.assertTrue(episodes[0].isdisjoint(episodes[1]))

    def test_missing_training_classes_rejected(self):
        with self.assertRaisesRegex(ValueError, '训练集每类'):
            split_sessions([{'label': 'rat', 'session_id': 'single'}])

    def test_raw_image_cannot_masquerade_as_difference(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            directory = root / 's1'
            directory.mkdir()
            for name, value in [('image', 100), ('raw', 100), ('baseline', 30)]:
                cv2.imencode('.png', np.full((32, 32, 3), value, np.uint8))[1].tofile(directory / f'{name}.png')
            row = {'image': 's1/image.png', 'raw_image': 's1/raw.png', 'baseline': 's1/baseline.png',
                   'session_id': 's1', 'episode_id': 'e1', 'label': 'rat', 'contact_detected': True, 'input_kind': INPUT_KIND}
            (directory / 'manifest.jsonl').write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '绝对差分图'):
                read_samples(root)


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_import_does_not_import_training_frameworks(self):
        script = "import core.inference, sys; assert not any(k == x or k.startswith(x+'.') for k in sys.modules for x in ('torch','torchvision','ultralytics'))"
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validated_onnx_runtime_and_mismatched_manifest(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        with temporary_directory() as tmp:
            model = Path(tmp) / 'test.onnx'
            p = distribution('rat')[None]
            embedding = np.array([[1, 0]], np.float32)
            graph = helper.make_graph([
                helper.make_node('Constant', [], ['probabilities'], value=numpy_helper.from_array(p)),
                helper.make_node('Constant', [], ['embedding'], value=numpy_helper.from_array(embedding)),
            ], 'contract-test', [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 32, 32])],
                [helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [1, 14]),
                 helper.make_tensor_value_info('embedding', TensorProto.FLOAT, [1, 2])])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9), model)
            manifest = fixture_manifest()
            manifest['sha256'] = file_sha256(model)
            manifest_path = model.with_suffix('.json')
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            classifier = OnnxClassifier(model)
            self.assertTrue(classifier.predict(np.zeros((32, 32, 3), np.uint8)).accepted)
            self.assertEqual(classifier.input_kind, INPUT_KIND)
            manifest['input_kind'] = 'unrecognized'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '预处理'):
                OnnxClassifier(model, preview=True)
            manifest['input_kind'] = 'raw_bgr_v1'
            manifest['status'] = 'unvalidated'
            manifest['thresholds']['rat'] = {'confidence': 1.0, 'margin': 1.0, 'distance': 0.0}
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '真实数据'):
                OnnxClassifier(model)
            preview = OnnxClassifier(model, preview=True)
            self.assertEqual(preview.input_kind, 'raw_bgr_v1')
            self.assertEqual(preview.predict(np.zeros((32, 32, 3), np.uint8)).reason, 'preview_top1')
            manifest['synthetic'] = True
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '真实数据'):
                OnnxClassifier(model, preview=True)
            manifest['status'] = 'validated'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '真实数据'):
                OnnxClassifier(model)
            manifest['synthetic'], manifest['sha256'] = False, 'wrong'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                OnnxClassifier(model, preview=True)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                OnnxClassifier(model)


if __name__ == '__main__':
    unittest.main()
