"""The live GUI must never retain a class after the inference gate rejects it."""

import unittest
import time
from dataclasses import replace
import wave
from unittest.mock import patch

import numpy as np

from gui.runtime import prepare_runtime

prepare_runtime()

from core.config import Config
from core.camera import FramePacket
from core.contact import ContactResult
from core.inference import ModelPrediction
from core.state_machine import RecognitionState
from gui.app import ASSETS_DIR, TouchSignView
from core.zodiac import ZODIAC_KEYS, ZODIAC_BY_KEY


class LiveViewTests(unittest.TestCase):
    def setUp(self):
        # No camera scan or camera acquisition is performed in GUI tests.
        with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as cameras:
            cameras.cameras = []
            self.view = TouchSignView(Config())

    def accepted_state(self):
        prediction = ModelPrediction('rat', 'rat', .96, .9, .02, True,
                                     'accepted', (('rat', .96), ('unknown', .04)), 11.0)
        return RecognitionState('recognized', '已确认生肖接触', ContactResult(
            calibrated=True, active=True, stable=True, score=.2, area_fraction=.3,
            duration_s=1.0, stability=1.0), prediction, True, 'REAL-RECORD')

    def test_startup_has_no_prediction_or_fabricated_history(self):
        self.assertIsNone(self.view.current)
        self.assertEqual(self.view.records, [])
        self.assertEqual(self.view.result.title.text, '未识别')
        self.assertTrue(self.view.difference_preview)
        self.assertEqual(self.view.sensor.live_text.text, 'NO INPUT')

    def test_speech_drops_changing_candidates_and_waits_for_release(self):
        rat = replace(self.accepted_state(), status='preview', new_record=False)
        tiger = replace(rat, prediction=replace(rat.prediction, key='tiger', predicted_key='tiger'))
        released = RecognitionState('no_contact', '未接触')
        samples = [(0, rat), (.4, tiger), (.8, rat), (1.2, rat), (1.61, rat),
                   (1.7, released), (1.9, released), (2, tiger), (2.4, tiger),
                   (2.8, tiger), (3, released), (3.3, released), (3.51, released),
                   (3.6, rat), (4, rat), (4.41, rat)]
        with patch('gui.app.ui.run_javascript') as play:
            for timestamp, state in samples:
                self.view._announce(state, timestamp)
                self.assertEqual(play.call_count, int(timestamp >= 1.61) + int(timestamp >= 4.41))
            self.assertTrue(all('/rat.wav' in call.args[0] for call in play.call_args_list))

    def test_new_stable_class_can_interrupt_without_release(self):
        from gui.speech import SpeechGate
        gate = SpeechGate(0.5)
        rat = self.accepted_state()
        self.assertEqual(gate.update(rat, 0), 'rat')
        tiger = replace(rat, status='preview', new_record=False,
                        prediction=replace(rat.prediction, key='tiger', predicted_key='tiger'))
        self.assertIsNone(gate.update(tiger, .1))
        self.assertIsNone(gate.update(tiger, .5))
        self.assertEqual(gate.update(tiger, .95), 'tiger')
        self.assertIsNone(gate.update(tiger, 1.1))

    def test_speech_skips_missing_frames_and_honors_mute(self):
        state = self.accepted_state()
        preview = replace(state, status='preview', new_record=False)
        with patch('gui.app.ui.run_javascript') as play:
            for timestamp in (0, .3, 1.5, 1.5, 1.9):
                self.view._announce(preview, timestamp)
            play.assert_not_called()
            self.view._announce(state, 2)
            self.view._announce(replace(state, new_record=False), 2.1)
            play.assert_called_once()
            self.view._toggle_speech()
            play.reset_mock()
            self.view._announce(state, 3)
            play.assert_not_called()

    def test_speech_test_updates_panel_without_creating_recognition(self):
        from types import SimpleNamespace
        for key in ZODIAC_KEYS:
            self.view._speech_test_changed(SimpleNamespace(args={'detail': key}))
            self.assertTrue(self.view.speech_testing)
            self.assertEqual(self.view.result.result_state_text.text, '语音试听')
            self.assertEqual(self.view.result.confidence.content, '—')
            self.assertEqual(self.view.result.title.text, ZODIAC_BY_KEY[key].title)
            self.view._render_state(self.accepted_state())
            self.assertEqual(self.view.records, [])
            self.assertEqual(self.view.result.result_state_text.text, '语音试听')
        self.view._speech_test_changed(SimpleNamespace(args={'detail': None}))
        self.assertFalse(self.view.speech_testing)
        self.assertEqual(self.view.result.title.text, '未识别')

    def test_every_zodiac_has_a_playable_clip(self):
        for key in ZODIAC_KEYS:
            with self.subTest(key=key), wave.open(str(ASSETS_DIR / 'speech' / f'{key}.wav')) as clip:
                self.assertGreater(clip.getnframes(), 0)
                self.assertGreater(len(set(clip.readframes(clip.getnframes()))), 1)

    def test_rejections_clear_previous_result_and_matrix(self):
        for status in ('unknown', 'no_contact', 'model_missing', 'no_signal', 'uncalibrated', 'error'):
            with self.subTest(status=status), patch('gui.app.ui.timer'):
                self.view._render_state(self.accepted_state())
                self.assertEqual(self.view.current.zodiac.key, 'rat')
                self.view._render_state(RecognitionState(status, '没有有效识别结果'))
                self.assertIsNone(self.view.current)
                self.assertEqual(self.view.result.title.text, '未识别')
                self.assertEqual(self.view.result.confidence.content, '—')
                for card in self.view.matrix.cards.values():
                    self.assertNotIn('is-active', card._classes)

    def test_history_only_updates_for_new_confirmed_record(self):
        from dataclasses import replace
        with patch('gui.app.ui.timer'):
            state = self.accepted_state()
            self.view._render_state(state)
            self.view._render_state(replace(state, new_record=False))
        self.assertEqual(len(self.view.records), 1)
        self.assertEqual(self.view.records[0].record_id, 'REAL-RECORD')

    def test_preview_shows_top_class_and_leaves_official_history_untouched(self):
        from dataclasses import replace
        state = replace(self.accepted_state(), status='preview', new_record=False, record_id='')
        with patch('gui.app.ui.timer'):
            self.view._render_state(state)
            self.assertEqual(self.view.current.zodiac.key, 'rat')
            self.assertEqual(self.view.result.result_state_text.text, '模型预览')
            self.assertIn('TOP 1', self.view.result.match_chip.content)
            for key, label in [('unknown', '未知物体'), ('no_contact', '空载')]:
                prediction = replace(state.prediction, key=None, predicted_key=key, accepted=False)
                self.view._render_state(replace(state, prediction=prediction))
                self.assertIsNone(self.view.current)
                self.assertEqual(self.view.result.title.text, label)
                self.assertIn('96.0', self.view.result.confidence.content)
                self.assertEqual(self.view.result.result_state_text.text, '模型预览')
                self.assertFalse(any('is-active' in card._classes for card in self.view.matrix.cards.values()))
            self.assertEqual(self.view.records, [])
            self.assertEqual(self.view.result.record_id.text, '')
            self.view._render_state(RecognitionState('no_contact', '未接触'))
        self.assertEqual(self.view.result.result_state_text.text, '无接触')
        self.assertEqual(self.view.result.confidence.content, '—')

    def test_cancelled_inference_during_shutdown_skips_rendering(self):
        async def cancelled_io(*args, **kwargs):
            return None

        self.view.streaming = True
        self.view._pending_reset = False
        self.view._stream_serial = 1
        frame = np.full((64, 64, 3), 100, dtype=np.uint8)
        for fresh in (False, True):
            with self.subTest(fresh=fresh), \
                    patch('gui.app.run.io_bound', cancelled_io), \
                    patch('gui.app.camera_manager') as cameras, \
                    patch.object(self.view, '_render_state') as render_state, \
                    patch.object(self.view, '_render_preview') as render_preview, \
                    patch('gui.app.LOGGER.exception') as log_error:
                cameras.frame_packet.return_value = (
                    FramePacket(1, 1, time.monotonic(), frame) if fresh else None)
                coroutine = self.view._tick()
                try:
                    with self.assertRaises(StopIteration):
                        coroutine.send(None)
                finally:
                    coroutine.close()
                render_state.assert_not_called()
                render_preview.assert_not_called()
                log_error.assert_not_called()
                self.assertFalse(self.view.analyzing)
                self.assertFalse(self.view.signal_lost)

    def test_reconnect_clears_result_and_accepts_new_stream_after_recalibration(self):
        async def immediate_io(function, *args, **kwargs):
            return function(*args, **kwargs)

        def tick_without_socket_loop():
            # The fake worker completes immediately, so this coroutine must finish
            # without suspension. This avoids Windows asyncio's socketpair while
            # exercising the actual async tick/reconnect branch.
            coroutine = self.view._tick()
            try:
                with self.assertRaises(StopIteration):
                    coroutine.send(None)
            finally:
                coroutine.close()

        frame = np.full((64, 64, 3), 100, dtype=np.uint8)
        baseline_frames = [frame.copy() for _ in range(10)]
        old_serial, recovered_serial = 5, 1_000_005
        self.view.engine.calibrate(baseline_frames)
        self.view.streaming = True
        self.view._pending_reset = False
        self.view._stream_serial = old_serial
        with patch('gui.app.ui.timer'), patch('gui.app.run.io_bound', immediate_io), \
                patch('gui.app.camera_manager') as cameras:
            cameras.statistics.return_value = (20.0, 0.0)
            self.view._render_state(self.accepted_state())
            cameras.frame_packet.return_value = FramePacket(
                recovered_serial, 1, time.monotonic(), frame)
            tick_without_socket_loop()
            self.assertIsNone(self.view.current)
            self.assertIsNone(self.view.engine.detector.baseline)
            self.assertEqual(self.view._stream_serial, recovered_serial)
            self.assertEqual(self.view.result.result_state_text.text, '待校准')

            tick_without_socket_loop()
            self.assertEqual(self.view.engine.state.status, 'uncalibrated')
            self.assertEqual(self.view._last_token, (recovered_serial, 1))

            # Recalibration of the recovered stream must allow fresh frames to
            # proceed, rather than repeatedly treating that serial as foreign.
            self.view.engine.calibrate(baseline_frames)
            cameras.frame_packet.return_value = FramePacket(
                recovered_serial, 2, time.monotonic(), frame)
            tick_without_socket_loop()
            self.assertEqual(self.view.engine.state.status, 'no_contact')
            self.assertIsNotNone(self.view.engine.detector.baseline)
            self.assertEqual(self.view._last_token, (recovered_serial, 2))
            self.assertFalse(self.view.analyzing)


if __name__ == '__main__':
    unittest.main()
