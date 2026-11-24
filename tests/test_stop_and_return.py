"""The user stop button releases then returns; disconnect still aborts motion."""
import asyncio
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

from core.contact import ContactResult
from robot.service import RobotService
from robot.settings import RobotSettings, SettingsStore
from tests.test_robot_cycle import OfflineLoop


class StopAndReturnTests(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(loop_factory=OfflineLoop, debug=True)

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.owner = object()
        self.settings = RobotSettings(stable_seconds=.1, move_timeout=1,
                                      release_timeout=.5, recognize_timeout=2)
        self.s = RobotService(self.settings, SettingsStore(Path(self.temp.name)/'config.yaml'))
        await self.s.connect(self.owner)
        self.arm = self.s.arm
        self.settings.points['mock'] = {'detect':[.1]*7, 'home':[0.]*7}

    async def asyncTearDown(self):
        if self.s.arm:
            self.arm.fail_feedback = False
            await self.s.disconnect(self.owner)
        self.s.pool.shutdown(wait=True)
        self.temp.cleanup()

    async def until(self, condition, timeout=3):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                self.fail(f'condition timed out: {self.s.phase}: {self.s.message}')
            await asyncio.sleep(.01)

    async def feed(self, status='no_contact'):
        while True:
            self.s.observe(SimpleNamespace(status=status, message='', prediction=None,
                           contact=ContactResult(calibrated=True, valid=True,
                                                 active=status!='no_contact')), time.monotonic(), 7)
            await asyncio.sleep(.02)

    async def test_recognition_stop_opens_returns_and_allows_next_explicit_cycle(self):
        reports = Mock()
        cycle = self.s.start(self.owner, ready=True, on_result=reports)
        await self.until(lambda: self.s.phase == 'recognizing')
        self.arm.events.clear()
        await self.s.stop_and_return(self.owner)
        await cycle
        self.assertEqual(self.arm.events, ['hold', ('gripper', .06), ('move', (0.,)*7)])
        self.assertEqual(self.s.phase, 'complete')
        self.assertIn('已张开并返回收回点', self.s.message)
        self.assertEqual(self.s.completed, 0)
        self.assertFalse(self.s.busy or self.s.finishing or self.s.stop_requested)
        reports.assert_not_called()
        await asyncio.sleep(.1)
        self.assertEqual(self.s.completed, 0)
        await self.s.start(self.owner, ready=True)
        self.assertEqual(self.s.completed, 1)

    async def test_stop_during_preparation_or_approach_also_opens_before_return(self):
        for phase in ('preparing', 'approaching'):
            with self.subTest(phase=phase):
                self.arm.events.clear()
                cycle = self.s.start(self.owner, ready=True)
                await self.until(lambda: self.s.phase == phase)
                await self.s.stop_and_return(self.owner)
                await cycle
                self.assertEqual(self.s.phase, 'complete')
                self.assertEqual(self.arm.events[-2:], [('gripper', .06), ('move', (0.,)*7)])
                self.assertNotIn(('gripper', .02), self.arm.events)

    async def test_stop_while_closing_cancels_closing_target_before_opening(self):
        feeder = asyncio.create_task(self.feed())
        reports = Mock()
        try:
            with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False):
                cycle = self.s.start(self.owner, ready=True, stream_id=7,
                                     ready_check=lambda: True, on_result=reports)
                await self.until(lambda: any(isinstance(e, tuple) and e[0]=='gripper' and e[1]<.06
                                            for e in self.arm.events))
                self.arm.events.clear()
                await self.s.stop_and_return(self.owner, stream_id=7)
                await cycle
                self.assertEqual(self.s.phase, 'complete')
                kinds = [e[0] if isinstance(e, tuple) else e for e in self.arm.events]
                self.assertLess(kinds.index('hold_gripper'), kinds.index('gripper'))
                self.assertEqual(self.arm.events[-2:], [('gripper', .06), ('move', (0.,)*7)])
                reports.assert_not_called()
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def test_repeated_clicks_share_one_open_and_return(self):
        self.arm.width = .02
        await asyncio.gather(*(self.s.stop_and_return(self.owner) for _ in range(3)))
        self.assertEqual(self.arm.events.count(('gripper', .06)), 1)
        self.assertEqual(self.arm.events.count(('move', (0.,)*7)), 1)
        self.assertEqual(self.s.phase, 'complete')

    async def test_explicit_stop_can_return_after_fault_with_healthy_feedback(self):
        self.s.phase, self.s.message = 'fault', '闭合超时'
        self.s.stop_requested = True
        self.arm.width = .02
        self.arm.enable = Mock()
        await self.s.stop_and_return(self.owner)
        self.assertEqual(self.s.phase, 'complete')
        self.assertEqual(self.arm.events[-2:], [('gripper', .06), ('move', (0.,)*7)])
        self.arm.enable.assert_not_called()

    async def test_stuck_opening_never_returns(self):
        self.arm.width = .02
        self.arm.stuck_gripper = True
        with self.assertLogs('robot.service', level='ERROR'):
            await self.s.stop_and_return(self.owner)
        self.assertEqual(self.s.phase, 'fault')
        self.assertIn('等待夹爪张开', self.s.message)
        self.assertNotIn(('move', (0.,)*7), self.arm.events)
        self.assertIn(('hold_gripper', .02), self.arm.events)

    async def test_contact_not_released_blocks_return_even_with_open_jaw(self):
        feeder = asyncio.create_task(self.feed('unstable'))
        try:
            with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False), \
                    self.assertLogs('robot.service', level='ERROR'):
                await self.s.stop_and_return(self.owner, stream_id=7, ready_check=lambda: True)
            self.assertEqual(self.s.phase, 'fault')
            self.assertIn('尚未确认无接触', self.s.message)
            self.assertIn(('gripper', .06), self.arm.events)
            self.assertNotIn(('move', (0.,)*7), self.arm.events)
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def test_disconnect_during_release_cancels_return(self):
        with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False):
            finishing = asyncio.create_task(self.s.stop_and_return(self.owner, stream_id=7))
            await self.until(lambda: self.s.phase == 'releasing')
            await self.s.disconnect(self.owner)
            await finishing
        self.assertIsNone(self.s.arm)
        self.assertFalse(self.arm.connected)
        self.assertNotIn(('move', (0.,)*7), self.arm.events)
        self.assertIn('hold', self.arm.events)

    async def test_disconnect_before_open_wins_over_finish_request(self):
        entered, proceed = asyncio.Event(), asyncio.Event()
        original_call = self.s.call
        first = True
        async def delayed_hold(name, *args):
            nonlocal first
            if name == 'hold' and first:
                first = False
                entered.set()
                await proceed.wait()
            return await original_call(name, *args)
        self.s.call = AsyncMock(side_effect=delayed_hold)
        finishing = asyncio.create_task(self.s.stop_and_return(self.owner))
        await entered.wait()
        disconnecting = asyncio.create_task(self.s.disconnect(self.owner))
        await self.until(lambda: self.s._disconnect_lock.locked() and not self.s.finishing)
        with self.assertRaisesRegex(RuntimeError, '正在断开'):
            await self.s.stop_and_return(self.owner)
        proceed.set()
        await asyncio.gather(finishing, disconnecting)
        self.assertIsNone(self.s.arm)
        self.assertFalse(any(isinstance(event, tuple) and event[0] in ('gripper', 'move')
                             for event in self.arm.events))

    async def test_bad_feedback_prevents_open_and_return(self):
        self.arm.fail_feedback = True
        with self.assertLogs('robot.service', level='ERROR'):
            await self.s.stop_and_return(self.owner)
        self.assertEqual(self.s.phase, 'fault')
        self.assertNotIn(('gripper', .06), self.arm.events)
        self.assertNotIn(('move', (0.,)*7), self.arm.events)

    async def test_first_return_and_retry_wait_for_new_frames_instead_of_old_cache(self):
        self.assertEqual(self.s.cycle_id, 0)
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                old_state = SimpleNamespace(status='uncalibrated', message='旧校准结果')
                old_timestamp = time.monotonic() - 10
                self.s.observe(old_state, old_timestamp, 99)
                release_seen = None
                self.arm.events.clear()
                async def delayed_feed():
                    nonlocal release_seen
                    while True:
                        if self.s.phase == 'releasing':
                            release_seen = time.monotonic() if release_seen is None else release_seen
                            if time.monotonic() - release_seen < .12:
                                # Even a worker delivering an OLD result after
                                # the cache was cleared must not fault this run.
                                self.s.observe(old_state, old_timestamp, 99)
                                self.assertNotIn(('move', (0.,)*7), self.arm.events)
                            else:
                                self.s.observe(SimpleNamespace(status='no_contact'), time.monotonic(), 7)
                        await asyncio.sleep(.01)
                feeder = asyncio.create_task(delayed_feed())
                try:
                    with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False):
                        await self.s.stop_and_return(self.owner, stream_id=7, ready_check=lambda: True)
                    self.assertEqual(self.s.phase, 'complete')
                    self.assertIsNotNone(release_seen)
                    self.assertEqual(self.s.cycle_id, 0)
                    self.assertEqual(self.arm.events.count(('move', (0.,)*7)), 1)
                finally:
                    feeder.cancel()
                    await asyncio.gather(feeder, return_exceptions=True)

    async def test_old_cache_without_new_frames_times_out_and_never_returns(self):
        self.s.observe(SimpleNamespace(status='no_contact'), time.monotonic()-10, 7)
        with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False), \
                self.assertLogs('robot.service', level='ERROR'):
            await self.s.stop_and_return(self.owner, stream_id=7, ready_check=lambda: True)
        self.assertEqual(self.s.phase, 'fault')
        self.assertIn('松开确认超时', self.s.message)
        self.assertIn('等待新的触觉反馈', self.s.message)
        self.assertNotIn(('move', (0.,)*7), self.arm.events)

    async def test_new_frame_from_different_camera_still_blocks_return(self):
        async def wrong_camera():
            while True:
                self.s.observe(SimpleNamespace(status='no_contact'), time.monotonic(), 8)
                await asyncio.sleep(.01)
        feeder = asyncio.create_task(wrong_camera())
        try:
            with patch.object(RobotService, 'mock', new_callable=PropertyMock, return_value=False), \
                    self.assertLogs('robot.service', level='ERROR'):
                await self.s.stop_and_return(self.owner, stream_id=7, ready_check=lambda: True)
            self.assertEqual(self.s.phase, 'fault')
            self.assertIn('预期流 7，收到流 8', self.s.message)
            self.assertNotIn(('move', (0.,)*7), self.arm.events)
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def test_first_return_prepares_limits_before_open_and_move(self):
        prepared = False
        original_move, original_gripper = self.arm.move, self.arm.gripper
        def prepare(q):
            nonlocal prepared
            self.assertEqual(tuple(q), (0.,)*7)
            prepared = True
        def gripper(width):
            self.assertTrue(prepared)
            original_gripper(width)
        def move(q):
            self.assertTrue(prepared, 'new connection has no limits until prepared')
            original_move(q)
        self.arm.prepare_move = Mock(side_effect=prepare)
        self.arm.move, self.arm.gripper = move, gripper
        self.arm.enable = Mock()
        await self.s.stop_and_return(self.owner)
        self.assertEqual(self.s.phase, 'complete')
        self.arm.prepare_move.assert_called_once()
        self.arm.enable.assert_not_called()

    async def test_limit_preparation_failure_never_opens_or_returns(self):
        self.arm.prepare_move = Mock(side_effect=TimeoutError('无法读取关节 J1 限位'))
        with self.assertLogs('robot.service', level='ERROR'):
            await self.s.stop_and_return(self.owner)
        self.assertEqual(self.s.phase, 'fault')
        self.assertIn('无法读取关节 J1 限位', self.s.message)
        self.assertNotIn(('gripper', .06), self.arm.events)
        self.assertNotIn(('move', (0.,)*7), self.arm.events)

    async def test_disconnect_during_limit_query_cannot_resume_return(self):
        entered, proceed = asyncio.Event(), asyncio.Event()
        original_call = self.s.call
        async def delayed_query(name, *args):
            if name == 'prepare_move':
                entered.set()
                await proceed.wait()
            return await original_call(name, *args)
        self.s.call = AsyncMock(side_effect=delayed_query)
        finishing = asyncio.create_task(self.s.stop_and_return(self.owner))
        await entered.wait()
        disconnecting = asyncio.create_task(self.s.disconnect(self.owner))
        await self.until(lambda: not self.s.finishing)
        proceed.set()
        await asyncio.gather(finishing, disconnecting)
        self.assertIsNone(self.s.arm)
        self.assertNotIn(('gripper', .06), self.arm.events)
        self.assertNotIn(('move', (0.,)*7), self.arm.events)

    async def test_missing_home_teaching_and_other_owner_are_rejected(self):
        with self.assertRaises(RuntimeError):
            await self.s.stop_and_return(object())
        self.s.phase = 'teaching'
        with self.assertRaises(RuntimeError):
            await self.s.stop_and_return(self.owner)
        self.s.phase = 'idle'
        self.settings.points['mock'].pop('home')
        with self.assertRaisesRegex(RuntimeError, '保存收回点'):
            await self.s.stop_and_return(self.owner)
        self.assertEqual(self.arm.events, [])
