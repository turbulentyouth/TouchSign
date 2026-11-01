"""Behavioral safety tests for the physical contact / recognition state machine."""
from dataclasses import replace
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from core.config import Config, ContactConfig, load_config
from core.contact import ContactDetector, make_difference
from core.inference import INPUT_KIND, RAW_INPUT_KIND, ModelPrediction
from core.state_machine import RecognitionEngine


class FakeModel:
    input_kind = INPUT_KIND

    def __init__(self):
        self.calls = []
        self.accepted = True
        self.key = 'rat'

    def predict(self, image):
        self.calls.append(image.copy())
        return ModelPrediction(self.key if self.accepted else None, self.key, .99, .98,
                               .01 if self.accepted else .9, self.accepted,
                               'accepted' if self.accepted else 'out_of_distribution',
                               ((self.key, .99), ('unknown', .01)), 1.0)


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.full((96, 128, 3), 80, np.uint8)
        self.touch = self.empty.copy()
        self.touch[25:65, 35:85] = 170

    def test_signed_underflow_never_occurs(self):
        dark = self.empty - 20
        self.assertTrue(np.all(make_difference(dark, self.empty) == 20))

    def test_noise_and_full_image_change_are_not_contact(self):
        detector = ContactDetector()
        detector.calibrate([self.empty]*10)
        self.assertFalse(detector.process(self.empty+2, 1).active)
        result = detector.process(self.empty+60, 2)
        self.assertFalse(result.active)
        self.assertFalse(result.valid)

    def test_motion_during_calibration_invalidates_previous_baseline(self):
        detector = ContactDetector()
        detector.calibrate([self.empty]*10)
        with self.assertRaises(ValueError):
            detector.calibrate([self.empty]*5+[self.empty+30]*5)
        self.assertFalse(detector.calibrated)

    def test_release_and_resolution_change_clear_contact(self):
        detector = ContactDetector(ContactConfig(stable_frames=1, stable_seconds=0))
        detector.calibrate([self.empty]*10)
        detector.process(self.touch, 1)
        self.assertTrue(detector.process(self.touch, 1.1).stable)
        self.assertFalse(detector.process(self.empty, 1.2).active)
        self.assertFalse(detector.process(self.empty[:64], 1.3).calibrated)

    def test_missing_config_uses_real_defaults(self):
        config = load_config(Path(__file__).parent/'does-not-exist.yaml')
        self.assertEqual(config.sensor.scan_max_index, 6)
        self.assertIsInstance(config.app.title, str)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.full((96, 128, 3), 80, np.uint8)
        self.touch = self.empty.copy()
        self.touch[25:65, 35:85] = 170
        scratch = Path(__file__).resolve().parents[1]/'.cache'/'tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        config = Config(contact=ContactConfig(stable_frames=1, stable_seconds=0))
        config = replace(config, recognition=replace(config.recognition, confirmation_frames=2,
            model_path=str(Path(self.temp.name)/'absent.onnx'), log_path=str(Path(self.temp.name)/'log.csv')))
        self.engine = RecognitionEngine(config)
        self.model = FakeModel()
        self.engine.model = self.model
        self.sequence = 0
        self.stamp = 100.0

    def step(self, frame, *, stream_id=1):
        self.sequence += 1
        self.stamp += .1
        with patch('core.state_machine.time.monotonic', return_value=self.stamp):
            return self.engine.process(frame, sequence=self.sequence, timestamp=self.stamp, stream_id=stream_id)

    def calibrate(self):
        self.engine.calibrate([self.empty]*10)

    def recognize(self):
        result = None
        for _ in range(4):
            result = self.step(self.touch)
        self.assertEqual(result.status, 'recognized', result.message)
        return result

    def test_gate_runs_before_model_and_receives_exact_difference(self):
        self.assertEqual(self.step(self.touch).status, 'uncalibrated')
        self.assertEqual(len(self.model.calls), 0)
        self.calibrate()
        for _ in range(4):
            self.assertEqual(self.step(self.empty).status, 'no_contact')
        self.assertEqual(len(self.model.calls), 0)
        self.recognize()
        np.testing.assert_array_equal(self.model.calls[0], make_difference(self.touch, self.empty))

    def test_raw_model_receives_sensor_pixels_only_after_stable_contact(self):
        self.model.input_kind = RAW_INPUT_KIND
        self.calibrate()
        self.assertEqual(self.step(self.empty).status, 'no_contact')
        self.assertEqual(self.step(self.touch).status, 'unstable')
        self.assertFalse(self.model.calls)
        self.recognize()
        np.testing.assert_array_equal(self.model.calls[0], self.touch)

    def test_one_record_per_contact_and_release_clears_result(self):
        self.calibrate()
        self.recognize()
        for _ in range(4):
            self.assertFalse(self.step(self.touch).new_record)
        released = self.step(self.empty)
        self.assertIsNone(released.prediction)
        self.recognize()
        with Path(self.engine.config.recognition.log_path).open(encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]['record_id'], rows[1]['record_id'])

    def test_unknown_immediately_clears_previous_recognition(self):
        self.calibrate()
        self.recognize()
        self.model.accepted = False
        result = self.step(self.touch)
        self.assertEqual(result.status, 'unknown')
        self.assertIsNone(result.prediction.key)
        self.assertFalse(result.new_record)

    def test_preview_shows_first_stable_frame_and_never_records(self):
        config = self.engine.config
        config = replace(config, recognition=replace(config.recognition, preview=True, confidence_threshold=1.0))
        with patch('core.state_machine.OnnxClassifier', return_value=self.model) as load:
            self.engine = RecognitionEngine(config)
        load.assert_called_once_with(self.engine.model_path, None, preview=True)
        self.assertIn('模型预览 · 未启用拒识', self.engine.model_status)
        self.assertEqual(self.step(self.touch).status, 'uncalibrated')
        self.calibrate()
        self.assertEqual(self.step(self.empty).status, 'no_contact')
        self.assertEqual(self.step(self.touch).status, 'unstable')
        self.assertEqual(len(self.model.calls), 0)
        state = self.step(self.touch)
        self.assertEqual((state.status, state.prediction.key), ('preview', 'rat'))
        self.assertFalse(state.new_record)
        self.assertEqual(state.record_id, '')
        for label in ('ox', 'unknown', 'no_contact'):
            self.model.key = label
            self.model.accepted = label == 'ox'
            state = self.step(self.touch)
            self.assertEqual((state.status, state.prediction.predicted_key), ('preview', label))
            self.assertEqual(state.prediction.key, label if self.model.accepted else None)
        self.assertIsNone(self.step(self.empty).prediction)
        self.assertFalse(Path(config.recognition.log_path).exists())

    def test_switch_out_of_preview_reapplies_confirmation_and_handles_failed_load(self):
        self.calibrate()
        with patch('core.state_machine.OnnxClassifier', return_value=self.model) as load:
            self.engine.reload_model(preview=True)
            self.engine.reload_model()
            load.assert_called_with(self.engine.model_path, None, preview=True)
            self.step(self.touch)
            self.assertEqual(self.step(self.touch).status, 'preview')
            self.engine.reload_model(preview=False)
            load.assert_called_with(self.engine.model_path, None, preview=False)
        self.assertFalse(self.engine.preview)
        self.assertEqual(self.step(self.touch).status, 'unstable')
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.assertTrue(self.step(self.touch).new_record)
        with patch('core.state_machine.OnnxClassifier', side_effect=ValueError('不匹配')):
            self.engine.reload_model(preview=True)
        self.assertIn('未就绪', self.engine.model_status)
        self.assertIsNone(self.engine.model)
        self.step(self.touch)
        self.assertEqual(self.step(self.touch).status, 'model_missing')

    def test_duplicate_frame_cannot_confirm_or_repeat_record(self):
        self.calibrate()
        self.step(self.touch)
        pending = self.step(self.touch)
        self.assertEqual(pending.status, 'confirming')
        for _ in range(5):
            with patch('core.state_machine.time.monotonic', return_value=self.stamp):
                repeated = self.engine.process(self.touch, sequence=self.sequence, timestamp=self.stamp, stream_id=1)
            self.assertEqual(repeated.status, 'confirming')
        self.assertEqual(len(self.model.calls), 1)

    def test_stale_frame_and_reconnect_require_new_baseline(self):
        self.calibrate()
        self.recognize()
        with patch('core.state_machine.time.monotonic', return_value=self.stamp+2):
            stale = self.engine.process(self.touch, sequence=self.sequence, timestamp=self.stamp, stream_id=1)
        self.assertEqual(stale.status, 'no_signal')
        self.assertIsNone(stale.prediction)
        self.assertFalse(self.engine.detector.calibrated)
        self.calibrate()
        self.step(self.touch)
        self.assertEqual(self.step(self.touch, stream_id=2).status, 'uncalibrated')

    def test_missing_model_does_not_invent_prediction(self):
        self.calibrate()
        self.engine.model = None
        for _ in range(4):
            result = self.step(self.touch)
        self.assertEqual(result.status, 'model_missing')
        self.assertIsNone(result.prediction)

    def test_different_class_during_same_touch_requires_release(self):
        self.calibrate()
        self.recognize()
        self.model.key = 'ox'
        for _ in range(4):
            result = self.step(self.touch)
        self.assertEqual(result.status, 'unknown')
        self.assertIsNone(result.prediction)
        self.step(self.empty)
        self.recognize()


if __name__ == '__main__':
    unittest.main()
