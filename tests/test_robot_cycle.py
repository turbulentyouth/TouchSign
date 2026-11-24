import asyncio
from pathlib import Path
import tempfile
import time
import unittest
import selectors
from types import SimpleNamespace
from robot.settings import RobotSettings, SettingsStore, joints
from robot.service import RobotService
from robot.backend import Snapshot


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
        self.assertEqual(self.s.phase,'fault')
        self.assertEqual(self.s.completed,0)
        self.assertNotIn(('move',(0.,)*7),self.s.arm.events)
        with self.assertRaises(RuntimeError):
            self.s.start(self.owner,ready=True)

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
    def test_invalid_numbers_and_points(self):
        for q in ([0]*6,[0]*6+[float('nan')],[0]*6+[9]):
            with self.assertRaises(ValueError): joints(q)
        for kwargs in ({'open_width':.02,'close_width':.05},{'gripper_force':float('nan')},
                       {'speed_percent':90},{'points':{'mock':{'via':[0]*7}}}):
            with self.assertRaises(ValueError): RobotSettings(**kwargs).validate()

    def test_simulated_points_are_not_hardware_points(self):
        settings=RobotSettings(points={'mock':{'detect':[0]*7,'home':[1]*7}})
        self.assertNotIn('nero:0:1.21',settings.points)
