"""The live GUI must never retain a class after the inference gate rejects it."""

import unittest
import time
from unittest.mock import patch

import numpy as np

from gui.runtime import prepare_runtime

prepare_runtime()

from core.config import Config
from core.camera import FramePacket
from core.contact import ContactResult
from core.inference import ModelPrediction
from core.state_machine import RecognitionState
from gui.app import TouchSignView


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
