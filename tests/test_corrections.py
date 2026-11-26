"""Correction labels and predictions stay tied to the frozen, lossless frame."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.camera import FramePacket
from core.config import Config
from core.contact import ContactResult
from core.corrections import CorrectionRecorder
from core.inference import ModelPrediction, RAW_INPUT_KIND
from core.state_machine import RecognitionEngine, RecognitionState
from training.data import read_image, read_samples


def model_fixture():
    return SimpleNamespace(model_path=Path('/models/current.onnx'), manifest_path=Path('/models/current.json'),
                           input_kind=RAW_INPUT_KIND, preview=True,
                           manifest={'sha256': 'loaded-model-hash', 'input_kind': RAW_INPUT_KIND,
                                     'preprocess': 'letterbox-rgb-255-v1', 'image_size': 224})


def state_fixture():
    prediction = ModelPrediction('monkey', 'monkey', .92, .7, .1, True, 'preview_top1',
                                 (('monkey', .92), ('rooster', .06)), 4.)
    return RecognitionState('preview', '人工按压', ContactResult(
        calibrated=True, active=True, stable=True, center_x=.8, center_y=.2), prediction)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'corrections'
        self.recorder = CorrectionRecorder(self.root)
        self.model = model_fixture()
        self.state = state_fixture()
        self.frame = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)

    def observe(self, sequence=1, state=None, *, pixels=None, timestamp=None, generation=0):
        packet = FramePacket(3, sequence, time.monotonic() if timestamp is None else timestamp,
                             self.frame.copy() if pixels is None else pixels)
        self.recorder.observe(packet, state or self.state, generation=generation,
                              model=self.model, camera=None, threshold=.5)
        return packet

    def freeze(self):
        return self.recorder.freeze(generation=0, stream_serial=3)

    def test_frozen_pixels_prediction_and_correct_label_round_trip_to_training(self):
        packet = self.observe()
        snapshot = self.freeze()
        packet.frame[:] = 0
        self.model.manifest['sha256'] = 'new-model-hash'
        self.observe(2, pixels=np.full_like(self.frame, 20))
        row = self.recorder.save(snapshot, 'rooster', '右上角，轻微倾斜')
        samples = read_samples(self.root, input_kind=RAW_INPUT_KIND, manual_grouping='frame')
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        np.testing.assert_array_equal(read_image(sample['path']), self.frame)
        self.assertEqual((sample['label'], sample['frame_sequence']), ('rooster', 1))
        detail = sample['correction']
        self.assertEqual(detail['prediction']['predicted_key'], 'monkey')
        self.assertEqual(detail['prediction']['confidence'], .92)
        self.assertEqual(detail['model']['sha256'], 'loaded-model-hash')
        self.assertEqual(detail['contact']['center_x'], .8)
        self.assertEqual(detail['notes'], '右上角，轻微倾斜')
        self.assertFalse(detail['prediction_matches_label'])
        self.assertTrue(Path(row['saved_path']).is_file())

    def test_label_is_required_and_same_frozen_frame_cannot_be_saved_twice(self):
        self.observe()
        snapshot = self.freeze()
        with self.assertRaisesRegex(ValueError, '正确生肖'):
            self.recorder.save(snapshot, None)
        self.assertFalse(self.root.exists())
        self.recorder.save(snapshot, 'rooster')
        with self.assertRaisesRegex(ValueError, '重复写入'):
            self.recorder.save(snapshot, 'rat')
        self.assertEqual(self.recorder.count, 1)
        self.assertEqual(len(list(self.root.rglob('*.png'))), 1)

    def test_low_confidence_is_savable_but_stale_or_changed_device_is_not(self):
        low = replace(self.state, status='unknown', prediction=replace(self.state.prediction, confidence=.4))
        self.observe(state=low)
        self.assertEqual(self.freeze().details['prediction']['confidence'], .4)
        for kwargs in ({'generation': 1, 'stream_serial': 3}, {'generation': 0, 'stream_serial': 4}):
            with self.assertRaisesRegex(ValueError, '新鲜'):
                self.recorder.freeze(**kwargs)
        self.observe(2, timestamp=time.monotonic() - 2)
        with self.assertRaisesRegex(ValueError, '新鲜'):
            self.freeze()
        self.observe(3, RecognitionState('no_contact', '已松开'))
        with self.assertRaisesRegex(ValueError, '新鲜'):
            self.freeze()

    def test_same_press_keeps_group_even_when_frame_grouping_is_requested(self):
        for sequence in (1, 2):
            self.observe(sequence, pixels=np.full_like(self.frame, sequence))
            self.recorder.save(self.freeze(), 'rooster')
        self.observe(3, RecognitionState('no_contact', '已松开'))
        self.observe(4, pixels=np.full_like(self.frame, 4))
        self.recorder.save(self.freeze(), 'rooster')
        rows = read_samples(self.root, input_kind=RAW_INPUT_KIND, manual_grouping='frame')
        self.assertEqual(rows[0]['episode_id'], rows[1]['episode_id'])
        self.assertNotEqual(rows[0]['episode_id'], rows[2]['episode_id'])
        session_rows = read_samples(self.root, input_kind=RAW_INPUT_KIND)
        self.assertEqual(len({r['episode_id'] for r in session_rows}), 1)

    def test_failed_disk_write_leaves_no_sample_and_can_retry_frozen_frame(self):
        self.observe()
        snapshot = self.freeze()
        with patch('core.manual_collection.os.fsync', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.recorder.save(snapshot, 'rooster')
        self.assertEqual(self.recorder.count, 0)
        self.assertEqual(list(self.root.rglob('*.png')), [])
        self.assertEqual((self.recorder.writer.session_dir / 'samples.jsonl').read_text(), '')
        self.recorder.save(snapshot, 'rooster')
        self.assertEqual(len(read_samples(self.root, input_kind=RAW_INPUT_KIND)), 1)


async def immediate_io(function, *args, **kwargs):
    return function(*args, **kwargs)


def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration:
        return
    finally:
        coroutine.close()
    raise AssertionError('Expected synchronous test worker')


class CorrectionGuiTests(unittest.TestCase):
    def setUp(self):
        from gui.runtime import prepare_runtime
        prepare_runtime()
        from gui.manual_testing import ManualTestView

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'corrections'
        config = Config()
        config = replace(config, recognition=replace(config.recognition, preview=True, confirm_preview=True))
        with patch('core.state_machine.OnnxClassifier', return_value=model_fixture()):
            engine = RecognitionEngine(config)
        engine.process = Mock(return_value=state_fixture())
        with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as camera:
            camera.cameras = []
            self.view = ManualTestView(config, engine=engine, output=self.root)
        self.view.streaming = True
        self.view._stream_serial = 3
        self.view._pending_reset = False
        self.packet = FramePacket(3, 8, time.monotonic(), np.full((32, 48, 3), 80, np.uint8))

    def open_from_tick(self):
        with patch('gui.app.camera_manager') as camera, patch('gui.app.run.io_bound', immediate_io):
            camera.frame_packet.return_value = self.packet
            camera.statistics.return_value = (20., None)
            complete(self.view._tick())
        self.view._open_correction()

    def test_dialog_freezes_tick_frame_and_saves_human_label_after_live_view_changes(self):
        self.open_from_tick()
        self.assertTrue(self.view.correction_dialog.value)
        self.assertIsNone(self.view.correction_label.value)
        self.assertIn('猴', self.view.correction_prediction.text)
        self.view.correction_label.value = 'rooster'
        self.view.correction_notes.value = '右上角'
        new_packet = replace(self.packet, sequence=9, timestamp=time.monotonic(), frame=self.packet.frame + 1)
        self.view._on_processed_frame(new_packet, state_fixture())
        with patch('gui.manual_testing.run.io_bound', immediate_io), patch('gui.manual_testing.ui.notify'):
            complete(self.view._save_correction())
        row, = read_samples(self.root, input_kind=RAW_INPUT_KIND)
        self.assertEqual((row['label'], row['frame_sequence']), ('rooster', 8))
        np.testing.assert_array_equal(read_image(row['path']), self.packet.frame)
        self.assertFalse(self.view.correction_dialog.value)
        self.assertEqual(self.view.correction_count.text, '纠错样本：1 张')
        self.assertIsNone(self.view.robot_panel)

    def test_dialog_validation_failure_retry_and_cancel(self):
        self.open_from_tick()
        complete(self.view._save_correction())
        self.assertIn('请选择', self.view.correction_error.text)
        self.assertFalse(self.root.exists())
        self.view.correction_label.value = 'rooster'
        snapshot = self.view.correction_snapshot
        with patch('gui.manual_testing.run.io_bound', immediate_io), \
                patch.object(self.view.corrections, 'save', side_effect=OSError('disk full')):
            complete(self.view._save_correction())
        self.assertIs(self.view.correction_snapshot, snapshot)
        self.assertTrue(self.view.correction_save.enabled)
        self.assertIn('可重试', self.view.correction_error.text)
        self.view._cancel_correction()
        self.assertIsNone(self.view.correction_snapshot)
        self.assertFalse(self.view.correction_dialog.value)
        self.assertFalse(self.root.exists())

    def test_signal_loss_blocks_freezing_even_before_previous_frame_expires(self):
        self.view._on_processed_frame(self.packet, state_fixture())
        self.view.signal_lost = True
        with patch('gui.manual_testing.ui.notify') as notify:
            self.view._open_correction()
        self.assertIsNone(self.view.correction_snapshot)
        self.assertFalse(self.view.correction_dialog.value)
        notify.assert_called_once()

    def test_manual_desktop_routes_to_correction_view(self):
        from desktop_main import _run_ui
        from gui.manual_testing import ManualTestView
        args = SimpleNamespace(manual=True, self_test_report=None, no_show=True, port=8090)
        with patch('desktop_main.sys.platform', 'linux'), \
                patch('core.instance.desktop_instance', return_value=nullcontext(None)), \
                patch('gui.runtime.prepare_runtime'), patch('gui.app.run_app') as launch:
            _run_ui(Config(), args)
        self.assertIs(launch.call_args.kwargs['view_factory'], ManualTestView)


if __name__ == '__main__':
    unittest.main()
