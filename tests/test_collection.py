"""Collection safety and exact raw/reference/difference round-trip tests."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.camera import CameraInfo, FramePacket
from core.collection import CollectionController
from core.config import ContactConfig
from core.dataset import DatasetWriter, dataset_summary

TEST_TEMP = Path(__file__).resolve().parents[1] / '.cache' / 'tests-collection'
TEST_TEMP.mkdir(parents=True, exist_ok=True)


class FakeStream:
    serial = 1

    def __init__(self, frame):
        self.frame = frame
        self.sequence = 1
        self.timestamp = time.monotonic()

    def frame_packet(self):
        return FramePacket(self.serial, self.sequence, self.timestamp, self.frame.copy())

    def statistics(self):
        return 20.0, time.monotonic() - self.timestamp


class FakeCamera:
    def __init__(self, frame):
        self.stream = FakeStream(frame)


class DatasetTests(unittest.TestCase):
    def test_roundtrip_difference_and_raw_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            baseline = np.full((64, 64, 3), 100, dtype=np.uint8)
            frame = baseline.copy()
            frame[16:48, 16:48] = 30
            writer = DatasetWriter(folder)
            writer.start_session(baseline)
            row = writer.save(frame, label='rat', episode_id='press1', contact_score=.1, contact_area=.2, contact=True, stable=True)
            read = lambda name: cv2.imdecode(np.fromfile(Path(folder) / name, dtype=np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(read(row['image']), cv2.absdiff(frame, baseline))
            np.testing.assert_array_equal(read(row['raw_image']), frame)
            self.assertEqual(row['input_kind'], 'absdiff_bgr_v1')
            self.assertTrue(row['contact_detected'])
            line = json.loads((writer.session_dir / 'manifest.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(line, row)
            summary = dataset_summary(folder)
            self.assertEqual(summary['counts']['rat'], 1)
            self.assertEqual(summary['sessions_per_label']['rat'], 1)

    def test_labels_contact_state_and_size_are_checked(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            baseline = np.zeros((64, 64, 3), np.uint8)
            writer = DatasetWriter(folder)
            writer.start_session(baseline)
            for label, contact in [('rat', False), ('no_contact', True), ('../rat', True)]:
                with self.assertRaises(ValueError):
                    writer.save(baseline, label=label, episode_id='press', contact_score=0, contact_area=0, contact=contact)
            with self.assertRaises(ValueError):
                writer.save(baseline[:32], label='no_contact', episode_id='press', contact_score=0, contact_area=0, contact=False)
            self.assertFalse((writer.session_dir / 'manifest.jsonl').exists())


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        self.baseline = np.full((64, 64, 3), 80, np.uint8)
        self.camera = FakeCamera(self.baseline.copy())
        self.controller = CollectionController(self.camera, Path(self.directory.name), contact_config=ContactConfig(stable_frames=1, stable_seconds=0))
        self.controller.detector.calibrate([self.baseline] * 10)
        self.controller.writer.start_session(self.baseline)
        self.controller._selected_serial = 1

    def tearDown(self):
        self.directory.cleanup()

    def start(self, label):
        self.controller.choose_label(label)
        self.controller.start(confirmed=True, target=10, interval=.1, max_per_episode=3)

    def new_frame(self, frame):
        self.camera.stream.frame = frame
        self.camera.stream.sequence += 1
        self.camera.stream.timestamp += 0.05
        with patch('core.collection.time.monotonic', return_value=self.camera.stream.timestamp):
            return self.controller.poll()

    def test_preview_advances_after_release_even_when_empty_pixels_are_identical(self):
        pressed = self.baseline.copy()
        pressed[18:46, 18:46] = 160
        raw, difference = self.new_frame(pressed)
        np.testing.assert_array_equal(raw, pressed)
        np.testing.assert_array_equal(difference, cv2.absdiff(pressed, self.baseline))
        for _ in range(2):
            raw, difference = self.new_frame(self.baseline)
            np.testing.assert_array_equal(raw, self.baseline)
            np.testing.assert_array_equal(difference, np.zeros_like(self.baseline))

    def test_no_contact_cannot_be_saved_as_zodiac(self):
        self.start('rat')
        for _ in range(5):
            self.new_frame(self.baseline)
        self.assertEqual(self.controller.saved_run, 0)

    def test_global_image_change_cannot_be_saved_as_no_contact(self):
        self.start('no_contact')
        self.controller._inactive_since -= 1
        self.new_frame(np.full_like(self.baseline, 240))
        self.assertEqual(self.controller.saved_run, 0)

    def test_duplicate_frame_and_stale_source_never_saved_twice(self):
        self.start('no_contact')
        self.controller._inactive_since -= 1
        self.new_frame(self.baseline)
        self.assertEqual(self.controller.saved_run, 1)
        self.controller.poll()
        self.assertEqual(self.controller.saved_run, 1)
        self.camera.stream.timestamp -= 2
        self.controller.poll()
        self.assertFalse(self.controller.recording)
        self.assertIsNone(self.controller.detector.baseline)

    def test_stable_contact_saves_then_release_changes_episode(self):
        self.start('rat')
        contact = self.baseline.copy()
        contact[20:44, 20:44] = 150
        self.new_frame(contact)
        self.new_frame(contact)
        self.assertEqual(self.controller.saved_run, 1)
        first_episode = self.controller._episode_id
        self.new_frame(self.baseline)
        self.assertNotEqual(first_episode, self.controller._episode_id)
        self.assertEqual(self.controller.saved_run, 1)

    def test_label_switch_stops_recording(self):
        self.start('rat')
        self.controller.choose_label('ox')
        self.assertFalse(self.controller.recording)

    def test_repeated_single_shots_save_each_new_frame_of_one_steady_press(self):
        self.controller.choose_label('rat')
        self.controller.max_per_episode = 1
        contact = self.baseline.copy()
        contact[20:44, 20:44] = 150
        self.new_frame(contact)
        sequences = []
        for _ in range(3):
            self.new_frame(contact)  # New camera frame 50 ms later, with identical pixels.
            with patch('core.collection.time.monotonic', return_value=self.camera.stream.timestamp):
                self.assertTrue(self.controller.capture_once(expected_label='rat'))
                self.assertFalse(self.controller.capture_once(expected_label='rat'))  # Same sequence.
            sequences.append(self.camera.stream.sequence)
        self.assertFalse(self.controller.recording)
        self.new_frame(contact)  # Without another keypress there is no fourth sample.
        manifest = self.controller.writer.session_dir / 'manifest.jsonl'
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
        self.assertEqual([(r['label'], r['frame_sequence']) for r in rows], [('rat', seq) for seq in sequences])
        self.assertEqual(len({r['episode_id'] for r in rows}), 1)
        for key in ('image', 'raw_image'):
            paths = {self.controller.writer.root / row[key] for row in rows}
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))

    def test_repeated_single_shots_of_empty_surface_require_settled_release(self):
        self.controller.choose_label('no_contact')
        pressed = self.baseline.copy()
        pressed[20:44, 20:44] = 150
        with patch('core.collection.time.monotonic', side_effect=lambda: self.camera.stream.timestamp):
            self.new_frame(pressed)
            self.assertFalse(self.controller.capture_once())
            self.new_frame(self.baseline)
            self.assertFalse(self.controller.capture_once())
            self.camera.stream.timestamp += .31
            for _ in range(2):
                self.new_frame(self.baseline)
                self.assertTrue(self.controller.capture_once())
        self.assertEqual(self.controller.writer.counts['no_contact'], 2)

    def test_automatic_capture_still_limits_samples_per_press(self):
        self.controller.choose_label('rat')
        self.controller.start(confirmed=True, target=3, interval=.1, max_per_episode=1)
        contact = self.baseline.copy()
        contact[20:44, 20:44] = 150
        self.new_frame(contact)
        self.new_frame(contact)
        self.assertEqual(self.controller.saved_run, 1)
        contact[20:44, 20:44] = 160
        self.camera.stream.timestamp += .2
        self.new_frame(contact)
        self.assertTrue(self.controller.contact.stable)
        self.assertEqual(self.controller.saved_run, 1)

    def test_keypress_during_invalid_contact_is_not_deferred(self):
        self.controller.choose_label('unknown')
        self.assertFalse(self.controller.capture_once())
        contact = self.baseline.copy()
        contact[20:44, 20:44] = 150
        self.new_frame(contact)
        self.new_frame(contact)
        self.assertFalse(self.controller.recording)
        self.assertEqual(self.controller.writer.counts['unknown'], 0)

    def test_single_shot_checks_the_current_frame_and_never_relabels_one_press(self):
        self.controller.choose_label('rat')
        contact = self.baseline.copy()
        contact[20:44, 20:44] = 150
        self.new_frame(contact)
        self.camera.stream.sequence += 1
        self.camera.stream.timestamp += 0.05
        with patch('core.collection.time.monotonic', return_value=self.camera.stream.timestamp):
            self.assertFalse(self.controller.capture_once(expected_label='ox'))
            self.assertTrue(self.controller.capture_once(expected_label='rat'))
            self.controller.choose_label('ox')
            self.assertFalse(self.controller.capture_once())
        self.camera.stream.timestamp += 1
        self.new_frame(contact + 1)
        with patch('core.collection.time.monotonic', return_value=self.camera.stream.timestamp):
            self.assertFalse(self.controller.capture_once())
        self.assertEqual(self.controller.writer.counts['ox'], 0)

    def test_mode_switch_retains_duplicate_filter_and_rejects_stale_snapshot(self):
        self.controller.choose_label('no_contact')
        self.controller._inactive_since -= 1
        self.assertTrue(self.controller.capture_once())
        self.controller.start(confirmed=True, target=5, interval=.1, max_per_episode=3)
        self.assertFalse(self.controller.capture_once())
        self.camera.stream.timestamp += 2
        self.new_frame(self.baseline)
        self.controller.stop()
        self.assertEqual(self.controller.writer.counts['no_contact'], 1)
        self.camera.stream.timestamp = time.monotonic() - 1
        self.assertFalse(self.controller.capture_once())
        self.assertIsNone(self.controller.detector.baseline)

    def test_camera_reconnect_invalidates_previous_reference(self):
        self.start('rat')
        self.camera.stream.serial += 1_000_000
        self.new_frame(self.baseline)
        self.assertFalse(self.controller.recording)
        self.assertIsNone(self.controller.detector.baseline)

    def test_disconnect_requires_recalibration(self):
        self.start('rat')
        self.camera.stream = None
        self.controller.poll()
        self.assertFalse(self.controller.recording)
        self.assertIsNone(self.controller.detector.baseline)

    def test_empty_class_keeps_low_frequency_noise_but_rejects_exact_duplicates(self):
        self.start('no_contact')
        self.controller._inactive_since -= 1
        self.new_frame(self.baseline)
        changed = self.baseline.copy()
        changed[0, 0, 0] += 1
        self.new_frame(changed)
        self.assertEqual(self.controller.saved_run, 1)  # two-second minimum
        self.camera.stream.timestamp += 2
        self.new_frame(changed)
        self.assertEqual(self.controller.saved_run, 2)  # not rejected by MAE
        self.camera.stream.timestamp += 2
        self.new_frame(changed)
        self.assertEqual(self.controller.saved_run, 2)  # exact duplicate rejected

    def test_calibration_uses_configured_frame_count_and_rechecks_connection(self):
        self.controller.detector.config = ContactConfig(calibration_frames=12)
        self.camera.stream.camera = CameraInfo(0, 'test')
        count = [0]

        def packet():
            count[0] += 1
            return FramePacket(1, count[0], time.monotonic(), self.baseline.copy())

        with patch.object(self.camera.stream, 'frame_packet', side_effect=packet), patch('core.collection.time.sleep'):
            self.controller.calibrate(confirmed_unloaded=True)
        self.assertEqual(count[0], 12)
        count[0] = 0

        def reconnect_on_last_sleep(_seconds):
            if count[0] == 12:
                self.camera.stream.serial += 1_000_000

        with patch.object(self.camera.stream, 'frame_packet', side_effect=packet), patch('core.collection.time.sleep', side_effect=reconnect_on_last_sleep):
            with self.assertRaisesRegex(ValueError, '校准结束'):
                self.controller.calibrate(confirmed_unloaded=True)
        self.assertIsNone(self.controller.detector.baseline)


if __name__ == '__main__':
    unittest.main()
