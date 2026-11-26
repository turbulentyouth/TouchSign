"""Manual testing keeps the active model pipeline without creating an arm."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.config import Config, PROJECT_ROOT, load_config
from core.inference import RAW_INPUT_KIND
from core.state_machine import RecognitionEngine
from tests import test_recognition as recognition_fixtures


class ManualStartupTests(unittest.TestCase):
    def test_manual_entry_preserves_selected_model_contact_and_sensor_settings(self):
        from desktop_main import main

        config = load_config(PROJECT_ROOT / 'config/ubuntu/config.yaml')
        with tempfile.TemporaryDirectory() as tmp, \
                patch('core.paths.user_data_root', return_value=Path(tmp)), \
                patch('desktop_main.logging.basicConfig'), \
                patch('desktop_main.logging.FileHandler'), \
                patch('core.config.load_config', return_value=config), \
                patch('desktop_main._run_ui') as launch:
            self.assertEqual(main(['--manual', '--no-show', '--port', '8090']), 0)
        active, args = launch.call_args.args
        self.assertFalse(active.app.robot_controls)
        self.assertTrue(active.recognition.confirm_preview)
        self.assertEqual(active.recognition, replace(config.recognition, confirm_preview=True))
        self.assertEqual(active.sensor, config.sensor)
        self.assertEqual(active.contact, config.contact)
        self.assertTrue(args.no_show)
        self.assertEqual(args.port, 8090)

    def test_manual_entry_does_not_reopen_an_existing_robot_ui(self):
        from desktop_main import _run_ui

        args = SimpleNamespace(manual=True, self_test_report=None, no_show=False, port=None)
        with patch('desktop_main.sys.platform', 'linux'), \
                patch('core.instance.desktop_instance', return_value=nullcontext({'url': 'http://localhost:8080'})), \
                patch('gui.browser_launcher.open_browser') as browser, \
                patch('gui.app.run_app') as launch, patch('builtins.print') as message:
            _run_ui(Config(), args)
        launch.assert_not_called()
        browser.assert_not_called()
        self.assertIn('请先退出当前程序', message.call_args.args[0])


class ManualRecognitionTests(unittest.TestCase):
    step = recognition_fixtures.EngineTests.step
    calibrate = recognition_fixtures.EngineTests.calibrate

    def setUp(self):
        recognition_fixtures.EngineTests.setUp(self)
        config = replace(self.engine.config, recognition=replace(
            self.engine.config.recognition, preview=True, confirm_preview=True,
            confidence_threshold=.5))
        self.model.input_kind = RAW_INPUT_KIND
        self.confidence = .99
        predict = self.model.predict
        self.model.predict = Mock(side_effect=lambda frame: replace(predict(frame), confidence=self.confidence))
        with patch('core.state_machine.OnnxClassifier', return_value=self.model):
            self.engine = RecognitionEngine(config)
        self.calibrate()

    def test_repeated_manual_presses_use_raw_pixels_clear_on_release_and_do_not_write_official_logs(self):
        for _ in range(3):
            self.assertEqual(self.step(self.touch).status, 'unstable')
            self.assertEqual(self.step(self.touch).status, 'confirming')
            self.assertEqual(self.step(self.touch).status, 'confirming')
            state = self.step(self.touch)
            self.assertEqual((state.status, state.prediction.key), ('preview', 'rat'))
            self.assertFalse(state.new_record)
            np.testing.assert_array_equal(self.model.calls[-1], self.touch)
            released = self.step(self.empty)
            self.assertEqual(released.status, 'no_contact')
            self.assertIsNone(released.prediction)
        self.assertFalse(Path(self.engine.config.recognition.log_path).exists())

    def test_low_confidence_resets_confirmation_and_duplicate_frames_do_not_count(self):
        self.step(self.touch)
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.confidence = .499
        state = self.step(self.touch)
        self.assertEqual(state.status, 'unknown')
        self.assertIn('50%', state.message)
        self.confidence = .5
        self.assertEqual(self.step(self.touch).status, 'confirming')
        for _ in range(5):
            with patch('core.state_machine.time.monotonic', return_value=self.stamp):
                duplicate = self.engine.process(self.touch, sequence=self.sequence,
                                                timestamp=self.stamp, stream_id=1)
            self.assertEqual(duplicate.status, 'confirming')
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.assertEqual(self.step(self.touch).status, 'preview')

    def test_negative_and_interrupted_frames_require_new_confirmation(self):
        self.step(self.touch)
        self.step(self.touch)
        self.model.key, self.model.accepted = 'unknown', False
        state = self.step(self.touch)
        self.assertEqual((state.status, state.prediction.predicted_key), ('preview', 'unknown'))
        self.model.key, self.model.accepted = 'rat', True
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.stamp += .6
        self.assertEqual(self.step(self.touch).status, 'unstable')
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.model.key = 'ox'
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.assertEqual(self.step(self.touch).status, 'confirming')
        self.assertEqual(self.step(self.touch).prediction.key, 'ox')


if __name__ == '__main__':
    unittest.main()
