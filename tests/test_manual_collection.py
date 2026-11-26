"""Manual capture round trips using sensor packets, without hardware."""
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import collect_manual
from core.camera import CameraInfo, CameraStream, FramePacket
from core.manual_collection import ManualCollector, ManualDatasetWriter


class ManualCollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.writer = ManualDatasetWriter(self.directory.name)
        self.collector = ManualCollector(self.writer, interval=.2)
        self.addCleanup(self.collector.close)

    def packet(self, sequence=1, *, serial=1, stamp=None):
        return FramePacket(serial, sequence, time.monotonic() if stamp is None else stamp,
                           np.arange(18, dtype=np.uint8).reshape(2, 3, 3))

    def rows(self):
        return [json.loads(line) for line in
                (self.writer.session_dir / 'samples.jsonl').read_text().splitlines()]

    def finish_pending(self):
        deadline = time.monotonic() + 3
        while self.collector.pending and time.monotonic() < deadline:
            self.collector.poll(None)
            time.sleep(.005)
        self.assertEqual(self.collector.pending, 0)

    def test_undo_uses_capture_order_after_label_change_and_training_ignores_archive(self):
        from training.data import read_samples
        self.collector.choose_label('rat')
        self.collector.capture_once(self.packet(1))
        self.collector.choose_label('ox')
        self.collector.capture_once(self.packet(2))
        self.finish_pending()
        original_rows = self.rows()
        self.collector.choose_label('pig')
        self.assertTrue(self.collector.undo())
        self.finish_pending()
        self.assertEqual(self.collector.label, 'pig')
        self.assertEqual(self.collector.counts, {'rat': 1})
        self.assertEqual(self.rows(), original_rows[:1])
        undone = original_rows[-1]
        archive = self.writer.session_dir / '_undone' / undone['image']
        self.assertFalse((self.writer.session_dir / undone['image']).exists())
        np.testing.assert_array_equal(cv2.imread(str(archive)), self.packet().frame)
        self.assertEqual(json.loads(archive.with_suffix('.json').read_text())['sample'], undone)
        samples = read_samples(self.directory.name, input_kind='raw_bgr_v1', manual_grouping='frame')
        self.assertEqual([row['label'] for row in samples], ['rat'])

    def test_undo_queued_captures_stops_burst_and_supports_multiple_undo(self):
        release, entered = threading.Event(), threading.Event()
        original_save = self.writer.save

        def slow_save(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test worker timed out')
            return original_save(*args)

        with patch.object(self.writer, 'save', side_effect=slow_save):
            self.collector.choose_label('rat')
            self.collector.start(self.packet())
            try:
                for sequence in range(1, 5):
                    self.assertTrue(self.collector.capture_once(self.packet(sequence)))
                self.assertTrue(entered.wait(2))
                self.assertTrue(self.collector.undo())
                self.assertTrue(self.collector.undo())
                self.assertFalse(self.collector.recording)
                self.assertEqual(self.collector.pending, 6)
            finally:
                release.set()
            self.collector.close()
        self.assertEqual([row['frame_sequence'] for row in self.rows()], [1, 2])
        self.assertEqual(self.collector.counts, {'rat': 2})
        self.assertEqual(len(list((self.writer.session_dir / '_undone').rglob('*.png'))), 2)

    def test_undo_all_empty_undo_and_recapture_same_frame_with_correct_label(self):
        self.collector.undo()
        self.finish_pending()
        self.assertIn('没有可撤回', self.collector.message)
        self.assertIsNone(self.collector.error)
        self.collector.choose_label('rat')
        packet = self.packet()
        self.collector.capture_once(packet)
        self.collector.undo()
        self.finish_pending()
        self.assertEqual(self.rows(), [])
        self.assertEqual(sum(self.collector.counts.values()), 0)
        self.collector.choose_label('ox')
        packet = replace(packet, timestamp=time.monotonic())
        self.assertTrue(self.collector.capture_once(packet))
        self.collector.close()
        self.assertEqual(self.collector.counts, {'ox': 1})
        self.assertEqual([row['label'] for row in self.rows()], ['ox'])

    def test_failed_undo_manifest_replacement_preserves_dataset_and_counts(self):
        self.collector.choose_label('rat')
        self.collector.capture_once(self.packet())
        self.finish_pending()
        before = (self.writer.session_dir / 'samples.jsonl').read_bytes()
        image = self.writer.session_dir / self.rows()[0]['image']
        with patch('core.manual_collection.os.replace', side_effect=OSError('read-only disk')):
            self.collector.undo()
            self.finish_pending()
        self.assertEqual((self.writer.session_dir / 'samples.jsonl').read_bytes(), before)
        self.assertTrue(image.exists())
        self.assertEqual(self.collector.counts, {'rat': 1})
        self.assertIn('撤回失败', self.collector.error)
        self.assertEqual(list(self.writer.session_dir.glob('.samples-*.tmp')), [])

    def test_archive_failure_after_commit_still_excludes_sample_from_training(self):
        self.collector.choose_label('rat')
        self.collector.capture_once(self.packet())
        self.finish_pending()
        image = self.writer.session_dir / self.rows()[0]['image']
        with patch.object(Path, 'replace', side_effect=OSError('archive blocked')):
            self.collector.undo()
            self.finish_pending()
        self.assertEqual(self.rows(), [])
        self.assertEqual(sum(self.collector.counts.values()), 0)
        self.assertTrue(image.exists())
        self.assertIsNone(self.collector.error)
        self.assertIn('归档失败', self.collector.message)
        self.assertIsNone(self.writer.undo())

    def test_undo_refuses_externally_changed_manifest(self):
        self.writer.save(self.packet(), 'rat', 'now')
        manifest = self.writer.session_dir / 'samples.jsonl'
        manifest.write_text(manifest.read_text() + '{"external":true}\n')
        before = manifest.read_bytes()
        with self.assertRaisesRegex(RuntimeError, '其他程序修改'):
            self.writer.undo()
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(len(list((self.writer.session_dir / 'rat').glob('*.png'))), 1)

    def test_saved_pixels_rotate_raw_sensor_exactly_once_without_hud_or_resize(self):
        raw = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
        stream = CameraStream(CameraInfo(0, 'test'), encode_jpeg=False)
        stream._publish(raw)
        self.collector.choose_label('rat')
        self.assertTrue(self.collector.capture_once(stream.frame_packet()))
        self.collector.close()
        row, = self.rows()
        saved = cv2.imread(str(self.writer.session_dir / row['image']))
        np.testing.assert_array_equal(saved, cv2.rotate(raw, cv2.ROTATE_180))
        self.assertEqual(row['rotation_degrees'], 180)
        self.assertEqual(row['label'], 'rat')
        self.assertEqual(row['frame_shape'], list(raw.shape))
        self.assertEqual(self.collector.counts, {'rat': 1})

    def test_labels_are_required_but_calibration_and_contact_are_not(self):
        packet = self.packet()
        self.assertFalse(self.collector.capture_once(packet))
        self.collector.start(packet)
        self.assertFalse(self.collector.recording)
        self.collector.choose_label('no_contact')
        self.assertTrue(self.collector.capture_once(packet))
        self.collector.close()
        row, = self.rows()
        self.assertEqual(row['label'], 'no_contact')
        self.assertNotIn('contact_detected', row)
        self.assertNotIn('baseline', row)

    def test_stale_and_repeated_frames_are_not_saved(self):
        self.collector.choose_label('ox')
        self.assertFalse(self.collector.capture_once(None))
        self.assertFalse(self.collector.capture_once(self.packet(stamp=time.monotonic()-2)))
        packet = self.packet()
        self.assertTrue(self.collector.capture_once(packet))
        self.assertFalse(self.collector.capture_once(packet))
        self.collector.close()
        self.assertEqual(len(self.rows()), 1)

    def test_switching_label_stops_burst_and_queued_save_keeps_original_pixels_and_label(self):
        release, entered = threading.Event(), threading.Event()
        original_save = self.writer.save

        def slow_save(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test worker timed out')
            return original_save(*args)

        with patch.object(self.writer, 'save', side_effect=slow_save):
            self.collector.choose_label('rat')
            packet = self.packet()
            expected = packet.frame.copy()
            self.collector.start(packet)
            try:
                self.collector.poll(packet)
                self.assertTrue(entered.wait(2))
                self.collector.choose_label('ox')
                packet.frame[:] = 255
                self.assertFalse(self.collector.recording)
                self.assertEqual(sum(self.collector.counts.values()), 0)
            finally:
                release.set()
            self.collector.close()
        row, = self.rows()
        self.assertEqual(row['label'], 'rat')
        np.testing.assert_array_equal(cv2.imread(str(self.writer.session_dir / row['image'])), expected)

    def test_burst_respects_interval_and_does_not_catch_up_in_a_batch(self):
        self.collector.choose_label('pig')
        with patch('core.manual_collection.time.monotonic', return_value=100.):
            packet = self.packet(stamp=100.)
            self.collector.start(packet)
            self.collector.poll(packet)
        with patch('core.manual_collection.time.monotonic', return_value=100.1):
            self.collector.poll(self.packet(2, stamp=100.1))
        with patch('core.manual_collection.time.monotonic', return_value=101.):
            self.collector.poll(self.packet(3, stamp=101.))
        self.collector.close()
        self.assertEqual([row['frame_sequence'] for row in self.rows()], [1, 3])

    def test_missing_signal_and_reconnect_stop_burst(self):
        self.collector.choose_label('pig')
        packet = self.packet()
        self.collector.start(packet)
        self.assertIsNone(self.collector.poll(None))
        self.assertFalse(self.collector.recording)
        self.collector.start(packet)
        self.collector.poll(replace(packet, serial=1000001))
        self.assertFalse(self.collector.recording)
        self.assertEqual(sum(self.collector.counts.values()), 0)
        self.assertFalse((self.writer.session_dir / 'samples.jsonl').exists())

    def test_queue_is_bounded_and_close_finishes_accepted_saves(self):
        release = threading.Event()
        original_save = self.writer.save

        def slow_save(*args):
            if not release.wait(3):
                raise RuntimeError('test worker timed out')
            return original_save(*args)

        with patch.object(self.writer, 'save', side_effect=slow_save):
            self.collector.choose_label('unknown')
            try:
                for sequence in range(4):
                    self.assertTrue(self.collector.capture_once(self.packet(sequence)))
                self.assertFalse(self.collector.capture_once(self.packet(5)))
                self.assertEqual(self.collector.pending, 4)
            finally:
                release.set()
            self.collector.close()
        self.assertEqual(self.collector.pending, 0)
        self.assertEqual(self.collector.counts['unknown'], 4)
        self.assertEqual(len(list(self.writer.session_dir.rglob('*.png'))), 4)

    def test_failed_write_stops_capture_and_is_not_counted(self):
        with patch.object(self.writer, 'save', side_effect=OSError('disk full')):
            self.collector.choose_label('rat')
            self.collector.start(self.packet())
            self.collector.poll(self.packet())
            self.collector.close()
        self.assertFalse(self.collector.recording)
        self.assertIn('disk full', self.collector.error)
        self.assertEqual(sum(self.collector.counts.values()), 0)
        self.assertFalse(self.collector.capture_once(self.packet(2)))

    def test_failed_manifest_append_rolls_back_png_and_row(self):
        with patch('core.manual_collection.os.fsync', side_effect=[None, OSError('disk full')]):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.writer.save(self.packet(), 'rat', '2026-11-25T10:00:00+08:00')
        self.assertEqual(self.rows(), [])
        self.assertEqual(list(self.writer.session_dir.rglob('*.png')), [])

    def test_invalid_label_rejected_and_sessions_do_not_overwrite(self):
        with self.assertRaises(ValueError):
            self.writer.save(self.packet(), '../rat', 'now')
        with self.assertRaises(ValueError):
            self.collector.choose_label('../rat')
        second = ManualDatasetWriter(self.directory.name)
        self.assertNotEqual(second.session_dir, self.writer.session_dir)


class ManualWindowTests(unittest.TestCase):
    def test_undo_keyboard_shortcuts(self):
        collector = Mock()
        collector.poll.return_value = None
        collector.counts = {'rat': 0}
        collector.label = 'rat'
        canvas = Mock(size=(1100, 650))
        with patch.multiple(cv2, namedWindow=Mock(), resizeWindow=Mock(),
                            imshow=Mock(), destroyAllWindows=Mock(),
                            getWindowProperty=Mock(return_value=1),
                            waitKeyEx=Mock(side_effect=[ord('z'), ord('Z'), 8, 0xff08, ord('q')])):
            collect_manual.run_window(Mock(), collector, canvas, 7)
        self.assertEqual(collector.undo.call_count, 4)

    def test_keyboard_workflow_with_simulated_camera(self):
        """Use the real controller, image writer and Chinese renderer together."""
        with tempfile.TemporaryDirectory() as folder:
            writer = ManualDatasetWriter(folder)
            collector = ManualCollector(writer)
            self.addCleanup(collector.close)
            stream = CameraStream(CameraInfo(7, 'test'), encode_jpeg=False)
            raw = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)

            def next_frame():
                stream._publish(raw)
                return stream.frame_packet()

            camera = Mock(frame_packet=Mock(side_effect=next_frame))
            canvas = collect_manual.ManualCanvas()
            with patch.multiple(cv2, namedWindow=Mock(), resizeWindow=Mock(),
                                imshow=Mock(), destroyAllWindows=Mock(),
                                getWindowProperty=Mock(return_value=1),
                                waitKeyEx=Mock(side_effect=[ord('2'), ord(' '), ord('z'),
                                    ord('N'), ord(' '), 0xff08, ord('2'), ord(' '),
                                    ord('N'), ord(' '), ord('q')])):
                collect_manual.run_window(camera, collector, canvas, 7)
            collector.close()
            self.assertEqual(collector.counts, {'ox': 1, 'no_contact': 1})
            self.assertEqual(len(list((writer.session_dir / '_undone').rglob('*.png'))), 2)
            for image in writer.session_dir.rglob('*.png'):
                np.testing.assert_array_equal(cv2.imread(str(image)), cv2.rotate(raw, cv2.ROTATE_180))

    def test_entry_auto_selects_vbts_and_releases_camera_after_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selected = CameraInfo(8, 'VBTS test')
            stream = CameraStream(selected, encode_jpeg=False)

            def next_frame():
                stream._publish(np.zeros((20, 30, 3), np.uint8))
                return stream.frame_packet()

            camera = Mock()
            camera.scan.return_value = [CameraInfo(0, 'webcam'), selected]
            camera.select.return_value.wait_ready.return_value = True
            camera.frame_packet.side_effect = next_frame
            with patch.dict('os.environ', {'DISPLAY': ':test'}), \
                    patch('collect_manual.user_data_root', return_value=root / 'state'), \
                    patch('collect_manual.CameraManager', return_value=camera), \
                    patch.multiple(cv2, namedWindow=Mock(), resizeWindow=Mock(), imshow=Mock(),
                                   destroyAllWindows=Mock(), getWindowProperty=Mock(return_value=1),
                                   waitKeyEx=Mock(side_effect=[ord(' '), ord('q')])), \
                    redirect_stdout(io.StringIO()):
                result = collect_manual.main(['--output', str(root / 'data'), '--label', 'rat'])
            self.assertEqual(result, 0)
            camera.select.assert_called_once_with(selected, target_fps=20., encode_jpeg=False)
            camera.close.assert_called_once()
            self.assertEqual(len(list((root / 'data').glob('*/rat/*.png'))), 1)

    def test_existing_desktop_process_prevents_camera_access(self):
        from contextlib import redirect_stderr
        from core.instance import desktop_instance

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with desktop_instance(root / 'desktop.lock', 'http://127.0.0.1:8080'), \
                    patch('collect_manual.user_data_root', return_value=root), \
                    patch('collect_manual.CameraManager') as manager, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(collect_manual.main(['--list-cameras']), 1)
                manager.assert_not_called()

    def test_arguments_support_one_command_preselection(self):
        args = collect_manual.parse_args(['--camera', '8', '--label', 'rat', '--interval', '.1'])
        self.assertEqual(args.camera, 8)
        self.assertEqual(args.label, 'rat')
        self.assertEqual(args.interval, .1)


if __name__ == '__main__':
    unittest.main()
