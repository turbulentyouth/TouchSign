import asyncio
from pathlib import Path
import tempfile
import time
import unittest
import selectors
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, PropertyMock, patch
from robot.settings import RobotSettings, SettingsStore, joints
from robot.service import RobotService
from robot.backend import ArmNeedsEnable, Snapshot


class OfflineSelector(selectors.SelectSelector):
    """No network I/O in these tests; timers and executor futures still run."""
    def select(self, timeout=None):
        time.sleep(min(.01, timeout) if timeout is not None else .01)
        return []


class OfflineLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(OfflineSelector())
    def _make_self_pipe(self):
        pass
    def _close_self_pipe(self):
        pass
    def _write_to_self(self):
        pass


class CycleTests(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(loop_factory=OfflineLoop, debug=True)
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SettingsStore(Path(self.temp.name)/'config.yaml')
        self.settings = RobotSettings(stable_seconds=.1, move_timeout=1,
                                      release_timeout=.4, recognize_timeout=1.2)
        self.s = RobotService(self.settings, self.store)
        self.owner = object()
        await self.s.connect(self.owner)
        self.settings.points['mock'] = {'detect':[.1]*7, 'home':[0.]*7}

    async def asyncTearDown(self):
        if self.s.arm:
            self.s.arm.fail_feedback = False
            await self.s.disconnect(self.owner)
        self.s.pool.shutdown(wait=True)
        self.temp.cleanup()

    async def test_one_click_one_cycle_release_before_return(self):
        reports = []
        task = self.s.start(self.owner, ready=True, on_result=lambda *r:reports.append(r))
        with self.assertRaises(RuntimeError):
            self.s.start(self.owner, ready=True)
        await task
        self.assertEqual(self.s.phase, 'complete')
        self.assertEqual(self.s.completed, 1)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0], ('rat',None,True))
        events = self.s.arm.events
        self.assertEqual(events, [('gripper',.06),('move',(.1,)*7),('gripper',.02),
                                  ('gripper',.06),('move',(0.,)*7)])
        await asyncio.sleep(.2)
        self.assertEqual(self.s.completed, 1)

    async def test_initial_brake_does_not_latch_fault_and_teaching_can_enable(self):
        arm = self.s.arm
        snapshot = arm.snapshot
        arm.snapshot = Mock(side_effect=ArmNeedsEnable('等待使能'))
        def enable():
            arm.snapshot = snapshot
        arm.enable = Mock(side_effect=enable)
        await asyncio.sleep(.2)
        self.assertEqual(self.s.phase, 'disabled')
        self.assertFalse(self.s.stop_requested)
        arm.enable.assert_not_called()
        await self.s.teach(self.owner, True)
        arm.enable.assert_called_once()
        self.assertEqual(self.s.phase, 'teaching')

    async def test_failed_enable_never_opens_gripper_or_enters_teach(self):
        arm = self.s.arm
        arm.enable = Mock(side_effect=TimeoutError('抱闸仍未释放'))
        with self.assertLogs('robot.service', level='ERROR'), self.assertRaises(TimeoutError):
            await self.s.teach(self.owner, True)
        self.assertEqual(self.s.phase, 'fault')
        self.assertEqual(arm.events, [])
        with self.assertRaises(RuntimeError):
            await self.s.teach(self.owner, True)

    async def test_teaching_feedback_timeout_never_reports_success_or_saves_point(self):
        arm = self.s.arm
        def fail_after_command():
            arm.teaching = True
            raise TimeoutError('模式未确认')
        arm.enter_teach = Mock(side_effect=fail_after_command)
        with self.assertLogs('robot.service', level='ERROR'), self.assertRaises(TimeoutError):
            await self.s.teach(self.owner, True)
        self.assertEqual(self.s.phase, 'fault')
        self.assertFalse(self.s.teaching)
        self.assertFalse(self.s.busy)
        self.assertIn('模式未确认', self.s.message)
        with self.assertRaises(RuntimeError):
            await self.s.save_point(self.owner, 'detect')
        self.assertNotIn('hold', arm.events)

    async def test_disconnect_releases_resources_when_hold_fails(self):
        arm = self.s.arm
        arm.hold = Mock(side_effect=RuntimeError('无法保持'))
        with self.assertLogs('robot.service', level='WARNING'):
            await asyncio.gather(self.s.disconnect(self.owner), self.s.disconnect(self.owner))
        self.assertFalse(arm.connected)
        self.assertIsNone(self.s.arm)
        self.assertIsNone(self.s.monitor_task)
        self.assertEqual(self.s.phase, 'disconnected')
        self.assertIn('保持指令失败', self.s.message)
        arm.hold.assert_called_once()

    async def test_timeout_releases_and_returns_without_fabricated_result(self):
        self.settings.recognize_timeout=.15
        await self.s.start(self.owner, ready=True, mock_key='timeout')
        self.assertEqual(self.s.phase,'complete')
        self.assertIsNone(self.s.result[0])
        self.assertEqual(self.s.arm.q,(0.,)*7)

    async def test_stuck_release_never_returns(self):
        def jam(*args):
            self.s.arm.stuck_gripper=True
        await self.s.start(self.owner, ready=True, on_result=jam)
        self.assertEqual(self.s.phase,'fault')
        self.assertNotIn(('move',(0.,)*7),self.s.arm.events)

    async def test_stop_cannot_restart_or_return_automatically(self):
        task=self.s.start(self.owner,ready=True)
        await asyncio.sleep(.15)
        await self.s.stop(self.owner)
        await task
        self.assertEqual(self.s.phase,'stopped')
        self.assertIn('收回已取消', self.s.message)
        self.assertEqual(self.s.completed,0)
        self.assertNotIn(('move',(0.,)*7),self.s.arm.events)
        with self.assertRaises(RuntimeError):
            self.s.start(self.owner,ready=True)
        with self.assertRaises(RuntimeError):
            await self.s.teach(self.owner, True)

    async def test_stop_hold_failure_remains_a_fault(self):
        self.s.arm.hold = Mock(side_effect=RuntimeError('反馈过期'))
        with self.assertLogs('robot.service', level='ERROR'), self.assertRaises(RuntimeError):
            await self.s.stop(self.owner)
        self.assertEqual(self.s.phase, 'fault')
        self.assertIn('保持当前位置失败', self.s.message)
        self.s.arm.hold = Mock()
        message = self.s.message
        await self.s.stop(self.owner)
        self.assertEqual(self.s.phase, 'fault')
        self.assertEqual(self.s.message, message)

    async def test_open_gripper_waits_for_contact_release_before_returning(self):
        await self.run_contact_release_case(clear_contact=True)

    async def test_open_gripper_with_contact_times_out_with_specific_reason(self):
        with self.assertLogs('robot.service', level='ERROR'):
            await self.run_contact_release_case(clear_contact=False)

    async def run_contact_release_case(self, *, clear_contact):
        # Use the real sensor-gating path with simulated arm feedback. Opening
        # the gripper alone must not authorize the return move.
        self.s._recognize = AsyncMock(return_value=('rat', None))
        self.s._grip = AsyncMock()  # This case exercises release gating only.
        release_seen = False
        release_started = None
        async def feed():
            nonlocal release_seen, release_started
            while True:
                status = 'preview'
                if self.s.phase == 'releasing':
                    if release_started is None:
                        release_started = time.monotonic()
                    if time.monotonic() - release_started < .15:
                        self.assertNotIn(('move', (0.,)*7), self.s.arm.events)
                        if self.s.arm.width == self.settings.open_width:
                            release_seen = True
                    elif clear_contact:
                        status = 'no_contact'
                self.s.observe(SimpleNamespace(status=status), time.monotonic(), 7)
                await asyncio.sleep(.02)
        feeder = asyncio.create_task(feed())
        try:
            with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False):
                await self.s.start(self.owner, ready=True, stream_id=7)
            self.assertTrue(release_seen)
            if clear_contact:
                self.assertEqual(self.s.phase, 'complete')
                self.assertIn(('move', (0.,)*7), self.s.arm.events)
            else:
                self.assertEqual(self.s.phase, 'fault')
                self.assertIn('松开确认超时', self.s.message)
                self.assertIn('尚未确认无接触', self.s.message)
                self.assertNotIn(('move', (0.,)*7), self.s.arm.events)
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def test_missing_points_sensor_and_other_window_rejected(self):
        with self.assertRaises(RuntimeError):
            self.s.start(object(),ready=True)
        with self.assertRaises(RuntimeError):
            self.s.start(self.owner,ready=False)
        self.settings.points.clear()
        with self.assertRaises(RuntimeError):
            self.s.start(self.owner,ready=True)
        self.assertFalse(self.s.busy)
        self.assertEqual(self.s.arm.events,[])

    async def test_teach_save_to_shared_config_and_hold_current(self):
        self.store.path.write_text('app:\n  title: Existing\n',encoding='utf-8')
        await self.s.teach(self.owner,True)
        with self.assertRaises(RuntimeError):
            self.s.start(self.owner,ready=True)
        await self.s.call('simulate_drag',[.2]*7)
        await self.s.save_point(self.owner,'detect')
        await self.s.teach(self.owner,False)
        self.assertEqual(self.s.arm.q,(.2,)*7)
        self.assertIn('Existing',self.store.path.read_text())
        self.assertEqual(self.store.load().points['mock']['detect'],[.2]*7)
        self.assertEqual([p.name for p in self.store.path.parent.iterdir()],['config.yaml'])
        with self.assertRaises(RuntimeError):
            await self.s.save_point(self.owner,'home')

    async def test_disconnect_during_cycle_is_terminal(self):
        self.s.start(self.owner,ready=True)
        await asyncio.sleep(.1)
        await self.s.disconnect(self.owner)
        self.assertIsNone(self.s.arm)
        self.assertEqual(self.s.phase,'disconnected')
        self.assertEqual(self.s.completed,0)

    async def test_feedback_loss_does_not_issue_return(self):
        task=self.s.start(self.owner,ready=True)
        await asyncio.sleep(.2)
        self.s.arm.fail_feedback=True
        await task
        self.assertEqual(self.s.phase,'fault')
        self.assertNotIn(('move',(0.,)*7),self.s.arm.events)

    async def test_old_camera_observation_and_changed_stream_rejected(self):
        state=SimpleNamespace(status='no_contact')
        self.s.observe(state,time.monotonic()-1,1)
        with self.assertRaises(RuntimeError):
            self.s._observation(1)
        self.s.observe(state,time.monotonic(),2)
        with self.assertRaises(RuntimeError):
            self.s._observation(1)

    async def test_stale_arm_snapshot_not_accepted_as_arrival(self):
        original=self.s.arm.snapshot
        self.s.arm.snapshot=lambda: Snapshot((0.,)*7,.06,False,time.monotonic()-2)
        with self.assertRaises(RuntimeError):
            await self.s.wait_for(lambda s:True,.3)
        self.s.arm.snapshot=original


class SettingsTests(unittest.TestCase):
    def test_closing_speed_range_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory)/'config.yaml')
            for speed in (.001, .005, .010, .011, .020, .030):
                with self.subTest(speed=speed):
                    settings = RobotSettings(gripper_close_speed=speed)
                    store.save(settings)
                    self.assertEqual(store.load().gripper_close_speed, speed)

    def test_legacy_step_is_ignored_without_changing_target_or_points(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory)/'config.yaml')
            store.path.write_text('robot:\n  close_width: 0.0035\n  grip_step: 0.003\n'
                                  '  points:\n    mock:\n      detect: [0,0,0,0,0,0,0]\n'
                                  '      home: [1,1,1,1,1,1,1]\n', encoding='utf-8')
            settings = store.load()
            self.assertFalse(hasattr(settings, 'grip_step'))
            self.assertEqual(settings.gripper_close_speed, .005)
            self.assertEqual(settings.points, {'mock': {'detect':[0]*7, 'home':[1]*7}})
            store.save(settings)
            loaded = store.load()
            self.assertEqual(loaded.gripper_close_speed, .005)
            self.assertNotIn('grip_step', store.path.read_text())
            self.assertEqual(loaded.close_width, .0035)
            self.assertEqual(loaded.points, settings.points)

    def test_invalid_numbers_and_points(self):
        for q in ([0]*6,[0]*6+[float('nan')],[0]*6+[9]):
            with self.assertRaises(ValueError): joints(q)
        for kwargs in ({'open_width':.02,'close_width':.05},{'gripper_force':float('nan')},
                       {'speed_percent':90},{'points':{'mock':{'via':[0]*7}}},
                       {'close_width':0}, {'close_width':.0000001},
                       {'gripper_close_speed':0}, {'gripper_close_speed':.000999},
                       {'gripper_close_speed':.030001},
                       {'gripper_close_speed':float('nan')}, {'gripper_close_speed':float('inf')},
                       {'grip_timeout':0}, {'grip_timeout':float('inf')}):
            with self.assertRaises(ValueError): RobotSettings(**kwargs).validate()

    def test_simulated_points_are_not_hardware_points(self):
        settings=RobotSettings(points={'mock':{'detect':[0]*7,'home':[1]*7}})
        self.assertNotIn('nero:0:1.21',settings.points)
