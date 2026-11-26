"""Independent collection entry, exact-frame labels, GUI gates and robot cycles."""
from contextlib import nullcontext
from dataclasses import replace
import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from core.collection import CollectionController
from core.camera import FramePacket
from core.collection_engine import CollectionEngine
from core.config import Config, ContactConfig
from core.inference import RAW_INPUT_KIND
from core.state_machine import RecognitionState
from robot.collection_service import CollectionRobotService
from robot.settings import RobotSettings, SettingsStore
from tests.test_collection import FakeCamera
from training.data import read_samples


class CollectionFixture:
    def setup_collection(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = np.full((64, 64, 3), 80, np.uint8)
        self.pressed = self.baseline.copy()
        self.pressed[20:44, 20:44] = 150
        self.camera = FakeCamera(self.baseline)
        self.collector = CollectionController(self.camera, self.root / 'data',
                                             contact_config=ContactConfig(stable_frames=1, stable_seconds=0))
        self.collector.detector.calibrate([self.baseline] * 10)
        self.collector.writer.start_session(self.baseline)
        self.collector._selected_serial = self.camera.stream.serial
        self.collector.choose_label('unknown')
        self.engine = CollectionEngine(self.collector)

    def frame(self, pressed=True):
        stream = self.camera.stream
        stream.frame = self.pressed.copy() if pressed else self.baseline.copy()
        stream.sequence += 1
        stream.timestamp = time.monotonic()
        packet = stream.frame_packet()
        state = self.engine.process(packet.frame, sequence=packet.sequence,
                                    timestamp=packet.timestamp, stream_id=packet.serial)
        return packet, state


class CollectionEngineTests(CollectionFixture, unittest.TestCase):
    def setUp(self):
        self.setup_collection()

    def test_exact_frame_manual_label_and_training_round_trip(self):
        self.assertIsNone(self.engine.model)
        self.collector.start(confirmed=True, target=1, interval=.1, max_per_episode=1)
        self.frame()
        # The camera can advance while the GUI's previously obtained frame is processed.
        stream = self.camera.stream
        stream.frame = self.baseline.copy()
        stream.sequence += 10
        stamp = time.monotonic()
        self.engine.process(self.pressed, sequence=3, timestamp=stamp, stream_id=stream.serial)
        self.assertEqual(self.collector.saved_run, 1)
        self.assertFalse(self.collector.recording)
        for kind in ('absdiff_bgr_v1', RAW_INPUT_KIND):
            sample, = read_samples(self.collector.writer.root, input_kind=kind)
            self.assertEqual(sample['frame_sequence'], 3)
            self.assertEqual(sample['label'], 'unknown')
            self.assertTrue(sample['stable'])
        self.engine.process(self.pressed, sequence=3, timestamp=stamp, stream_id=stream.serial)
        self.assertEqual(self.collector.writer.counts['unknown'], 1)

    def test_disconnect_reconnect_and_stale_packets_invalidate_labels(self):
        for change in ('stale', 'serial', 'stream_recovered'):
            with self.subTest(change=change):
                self.collector.detector.calibrate([self.baseline] * 10)
                self.collector._selected_serial = self.camera.stream.serial
                self.collector.choose_label('rat')
                serial, stamp = self.camera.stream.serial, time.monotonic()
                if change == 'stale':
                    stamp -= 2
                elif change == 'serial':
                    serial += 1
                else:
                    self.camera.stream.serial += 1
                self.engine.process(self.pressed, sequence=9, timestamp=stamp, stream_id=serial)
                self.assertFalse(self.collector.detector.calibrated)
                self.assertIsNone(self.collector.label)
                self.assertFalse(self.collector.writer.counts)

    def test_new_robot_episode_requires_new_stability_after_manual_capture(self):
        self.frame()
        self.frame()
        self.assertTrue(self.collector.capture_once())
        previous_episode = self.collector._episode_id
        self.collector.start(confirmed=True, target=1, interval=.1, max_per_episode=1, new_episode=True)
        self.assertFalse(self.collector.contact)
        self.assertNotEqual(self.collector._episode_id, previous_episode)
        self.frame()
        self.assertEqual(self.collector.saved_run, 0)
        self.collector._last_saved_at -= .2
        self.frame()
        self.assertEqual(self.collector.saved_run, 1)
        self.assertEqual(self.collector.writer.counts['unknown'], 2)


class CollectionCycleTests(CollectionFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.setup_collection()
        settings = RobotSettings(open_width=.022, close_width=.02, gripper_close_speed=.03,
                                 stable_seconds=.1, recognize_timeout=1.2)
        self.service = CollectionRobotService(settings, SettingsStore(self.root / 'settings.yaml'))
        self.owner = object()
        await self.service.connect(self.owner)
        settings.points['mock'] = {'detect': [.1] * 7, 'home': [0.] * 7}

    async def asyncTearDown(self):
        await self.service.disconnect(self.owner)
        self.service.pool.shutdown(wait=True)

    def start(self, **overrides):
        args = dict(collector=self.collector, target=1, interval=.1,
                    stream_id=1, ready_check=lambda: self.collector.detector.calibrated)
        args.update(overrides)
        return self.service.start_collection(self.owner, **args)

    async def drive_frames(self, *, timeout=False, disk_error=False, stop=False):
        s = self.service
        initial = self.collector.saved_run
        stopper = None
        while s.busy:
            # The device keeps delivering frames while vision waits for jaw arrival.
            self.camera.stream.timestamp = time.monotonic()
            self.camera.stream.sequence += 1
            phase = s.phase
            if phase in ('recognizing', 'releasing'):
                if disk_error and phase == 'recognizing':
                    self.collector.writer.save = Mock(side_effect=OSError('磁盘已满'))
                p, state = self.frame(pressed=phase == 'recognizing' and not timeout)
                s.observe(state, p.timestamp, p.serial)
                if stop and phase == 'recognizing':
                    stopper = asyncio.create_task(s.stop_and_return(self.owner, stream_id=1,
                                                                    ready_check=lambda: True))
                    stop = False
            else:
                self.assertFalse(self.collector.recording, phase)
                self.assertEqual(self.collector.saved_run, initial if phase != 'returning' else s.capture_count)
            await asyncio.sleep(.04)
        if stopper is not None:
            await stopper

    async def test_full_capture_cycle_keeps_motion_and_unknown_label(self):
        with patch.object(CollectionRobotService, 'mock', new_callable=PropertyMock, return_value=False):
            task = self.start()
            await asyncio.wait_for(self.drive_frames(), 5)
            await task
        s = self.service
        self.assertEqual(s.phase, 'complete', s.message)
        self.assertEqual(s.completed, 1)
        self.assertEqual(s.capture_count, 1)
        self.assertFalse(self.collector.recording)
        self.assertEqual(s.arm.events[-2:], [('gripper', .022), ('move', (0.,) * 7)])
        row, = read_samples(self.collector.writer.root)
        self.assertEqual(row['label'], 'unknown')
        self.assertEqual(s.result[0], 'unknown')

    async def test_mock_cycle_does_not_write_samples(self):
        await self.start()
        self.assertEqual(self.service.phase, 'complete')
        self.assertFalse(self.collector.writer.counts)
        self.assertIn('未写入', self.service.capture_message)

    async def test_missing_label_and_no_contact_rejected_before_motion(self):
        for label in (None, 'no_contact'):
            self.collector.choose_label(label)
            with self.assertRaises(ValueError):
                self.start()
            self.assertFalse(self.service.busy)
            self.assertEqual(self.service.arm.events, [])

    async def test_timeout_still_releases_and_returns_without_samples(self):
        with patch.object(CollectionRobotService, 'mock', new_callable=PropertyMock, return_value=False):
            task = self.start()
            await asyncio.wait_for(self.drive_frames(timeout=True), 5)
            await task
        self.assertEqual(self.service.phase, 'complete', self.service.message)
        self.assertEqual(self.service.capture_count, 0)
        self.assertFalse(self.collector.writer.counts)
        self.assertIn('未达目标', self.service.capture_message)

    async def test_disk_error_stops_capture_and_holds(self):
        with patch.object(CollectionRobotService, 'mock', new_callable=PropertyMock, return_value=False), \
                patch('robot.service.LOGGER.exception'):
            task = self.start()
            await asyncio.wait_for(self.drive_frames(disk_error=True), 5)
            await task
        self.assertEqual(self.service.phase, 'fault')
        self.assertIn('磁盘已满', self.service.message)
        self.assertEqual(self.service.arm.events[-1], 'hold')
        self.assertFalse(self.collector.recording)
        self.assertFalse(self.collector.writer.counts)

    async def test_stop_during_capture_releases_without_adding_samples(self):
        with patch.object(CollectionRobotService, 'mock', new_callable=PropertyMock, return_value=False):
            task = self.start()
            await asyncio.wait_for(self.drive_frames(stop=True), 5)
            await task
        self.assertEqual(self.service.phase, 'complete', self.service.message)
        self.assertEqual(self.service.completed, 0)
        self.assertFalse(self.collector.recording)
        self.assertFalse(self.collector.writer.counts)
        self.assertEqual(self.service.arm.events[-2:], [('gripper', .022), ('move', (0.,) * 7)])

    async def test_label_change_while_waiting_aborts_capture(self):
        with patch.object(CollectionRobotService, 'mock', new_callable=PropertyMock, return_value=False):
            self.service.collector = self.collector
            self.service.capture_label = 'unknown'
            self.service.capture_target = 1
            self.service.capture_interval = .1
            task = asyncio.create_task(self.service._recognize(1, 'rat'))
            await asyncio.sleep(.02)
            self.collector.choose_label('ox')
            with self.assertRaises(RuntimeError):
                await task
        self.assertFalse(self.collector.recording)
        self.assertFalse(self.collector.writer.counts)


class CollectionGuiTests(unittest.TestCase):
    def test_gui_processes_only_frames_after_arrival_and_during_release(self):
        from gui.runtime import prepare_runtime
        prepare_runtime()
        from gui.collection_app import CollectionView

        async def immediate_io(function, *args, **kwargs):
            return function(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            s = CollectionRobotService(RobotSettings(), SettingsStore(Path(tmp) / 'settings.yaml'))
            self.addCleanup(s.pool.shutdown, wait=True)
            config = Config()
            config = replace(config, app=replace(config.app, robot_controls=True))
            with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as camera, \
                    patch('gui.collection_app.camera_manager', camera), patch('gui.robot_panel._service', s):
                camera.cameras = []
                v = CollectionView(config, output=Path(tmp) / 'data')
                s.owner = v.robot_panel
                s.busy = True
                s.recognition_start = time.monotonic()
                v.connected = v.streaming = True
                v._pending_reset = False
                v._stream_serial = 1
                frame = np.zeros((64, 64, 3), np.uint8)
                process = Mock(return_value=RecognitionState('no_contact', '未接触'))
                with patch.object(CollectionRobotService, 'mock', new_callable=PropertyMock, return_value=False), \
                        patch('gui.collection_app.run.io_bound', immediate_io), \
                        patch.object(v.engine, 'process', process), patch.object(v, '_render_preview'), \
                        patch('gui.collection_app.LOGGER.exception') as error:
                    for i, (phase, stamp, calls) in enumerate([
                        ('approaching', time.monotonic(), 0),
                        ('gripping', time.monotonic(), 0),
                        ('recognizing', s.recognition_start - .01, 0),
                        ('recognizing', time.monotonic(), 1),
                        ('releasing', time.monotonic(), 2),
                        ('returning', time.monotonic(), 2),
                    ], 1):
                        s.phase = phase
                        camera.frame_packet.return_value = FramePacket(1, i, stamp, frame)
                        coroutine = v._tick()
                        with self.assertRaises(StopIteration):
                            coroutine.send(None)
                        coroutine.close()
                        self.assertEqual(process.call_count, calls, phase)
                    error.assert_not_called()

    def test_build_reuses_robot_and_label_selection_enables_capture(self):
        from gui.runtime import prepare_runtime
        prepare_runtime()
        from gui.collection_app import CollectionView
        from robot.backend import MockArm
        with tempfile.TemporaryDirectory() as tmp:
            service = CollectionRobotService(RobotSettings(), SettingsStore(Path(tmp) / 'settings.yaml'))
            self.addCleanup(service.pool.shutdown, wait=True)
            config = Config()
            config = replace(config, app=replace(config.app, robot_controls=True))
            with patch('gui.app.ui.timer'), patch('gui.app.camera_manager') as cameras, \
                    patch('gui.robot_panel._service', service), patch('core.state_machine.OnnxClassifier') as model:
                cameras.cameras = []
                view = CollectionView(config, output=Path(tmp) / 'data')
                model.assert_not_called()
                self.assertIs(view.robot_panel.service, service)
                self.assertEqual(view.detect_button.text, '采集（运行一轮）')
                self.assertFalse(view.detect_button.enabled)
                self.assertFalse(view.result.capture.enabled)
                view.collector.detector.calibrate([np.zeros((64, 64, 3), np.uint8)] * 10)
                service.arm = MockArm(service.settings)
                service.owner = view.robot_panel
                service.phase = 'idle'
                service.settings.points['mock'] = {'detect': [.1] * 7, 'home': [0.] * 7}
                for label in ('rat', 'ox'):
                    view.result.label.value = label
                    view.validate_label()
                    view.result.refresh()
                    view.robot_panel.refresh()
                    self.assertEqual(view.collector.label, label)
                    self.assertTrue(view.result.capture.enabled)
                    self.assertTrue(view.detect_button.enabled)
                view.result.label.value = None
                with self.assertRaises(ValueError):
                    view.validate_label()
                view.result.refresh()
                view.robot_panel.refresh()
                self.assertFalse(view.result.capture.enabled)
                self.assertFalse(view.detect_button.enabled)
                service.busy = True
                view.result.refresh()
                self.assertFalse(view.result.label.enabled)
                self.assertFalse(view.result.target.enabled)
                self.assertFalse(view.result.capture.enabled)
                service.owner = None
                view._on_disconnect()
                self.assertFalse(view.connected)
                self.assertIsNone(view.collector.label)
                self.assertTrue(view.cancel_calibration.is_set())

    def test_entry_lock_prevents_competing_camera_process(self):
        from collect_gui import main
        with tempfile.TemporaryDirectory() as tmp, \
                patch('core.paths.user_data_root', return_value=Path(tmp)), \
                patch('core.instance.desktop_instance', return_value=nullcontext({'pid': 123})), \
                patch('gui.app.run_app') as run:
            self.assertEqual(main(['--no-show']), 1)
            run.assert_not_called()

    def test_entry_passes_only_collection_factory_to_shared_runner(self):
        from collect_gui import main
        from gui.collection_app import CollectionView
        with tempfile.TemporaryDirectory() as tmp, \
                patch('core.paths.user_data_root', return_value=Path(tmp)), \
                patch('core.instance.desktop_instance', return_value=nullcontext(None)), \
                patch('gui.app.run_app') as run:
            self.assertEqual(main(['--no-show', '--output', tmp]), 0)
            self.assertIs(run.call_args.kwargs['view_factory'].func, CollectionView)
            self.assertEqual(run.call_args.kwargs['port'], 8081)
            self.assertFalse(run.call_args.kwargs['show'])


if __name__ == '__main__':
    unittest.main()
