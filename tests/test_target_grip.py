"""Slow closure to one final target, with synthetic time/telemetry and no hardware."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from core.contact import ContactResult
from robot.backend import Snapshot
from robot.service import RobotService, Stopped
from robot.settings import RobotSettings
from tests.test_robot_cycle import OfflineLoop


class TargetGripTests(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(loop_factory=OfflineLoop, debug=True)

    async def asyncSetUp(self):
        self.now = 10.
        self.width = .060
        self.q = (0.,)*7
        self.timestamp_offset = 0.
        self.teaching = False
        self.stuck = False
        self.events = []
        self.before_commands = []
        self.on_snapshot = None
        self.on_command = None
        self.settings = RobotSettings(close_width=.0035, grip_timeout=2., stable_seconds=.2)
        self.service = RobotService(self.settings, Mock())
        self.service.arm = Mock()  # Exercise the hardware branch without a device.
        self.service.call = AsyncMock(side_effect=self.io)
        self.service.ready_check = lambda: True
        self.clock_patch = patch('robot.service.time', SimpleNamespace(monotonic=lambda: self.now))
        self.async_patch = patch('robot.service.asyncio', SimpleNamespace(sleep=self.sleep))
        self.clock_patch.start()
        self.async_patch.start()

    async def asyncTearDown(self):
        self.async_patch.stop()
        self.clock_patch.stop()
        self.service.pool.shutdown(wait=True)

    async def sleep(self, seconds):
        self.now += seconds

    async def io(self, name, *args):
        if name == 'snapshot':
            if self.on_snapshot:
                self.on_snapshot()
            return Snapshot(self.q, self.width, self.teaching, self.now + self.timestamp_offset)
        if name == 'gripper':
            self.before_commands.append(self.width)
            self.events.append((name, args[0], self.now))
            if not self.stuck:
                self.width = args[0]
            if self.on_command:
                self.on_command(args[0])
        elif name == 'hold_gripper':
            self.events.append((name, self.width, self.now))
            return self.width
        elif name == 'move':
            self.q = tuple(args[0])
            self.events.append((name, self.q, self.now))
        elif name in ('hold', 'enable'):
            self.events.append((name, None, self.now))
        else:
            self.fail(f'unexpected operation {name}')

    def targets(self):
        return [width for name, width, _ in self.events if name == 'gripper']

    def assert_slow_targets(self, initial=.060, started=10.):
        commands = [e for e in self.events if e[0]=='gripper']
        self.assertGreater(len(commands), 1)
        previous, sent_at = initial, started
        for _, width, timestamp in commands:
            self.assertLess(width, previous)
            self.assertGreaterEqual(width, self.settings.close_width)
            self.assertLessEqual(previous-width, self.settings.gripper_close_speed*(timestamp-sent_at)+1e-10)
            self.assertLessEqual(previous-width, self.settings.gripper_close_speed*.05+1e-10)
            previous, sent_at = width, timestamp
        self.assertEqual(previous, self.settings.close_width)
        self.assertEqual(self.targets().count(self.settings.close_width), 1)

    async def test_slow_ramp_without_tactile_processing(self):
        with patch.object(self.service, '_observation', side_effect=AssertionError('不应检测触觉')):
            await self.service._grip()
        self.assert_slow_targets()
        self.assertTrue(all(e[0]=='gripper' for e in self.events))
        self.assertGreaterEqual(self.now - self.events[-1][2], self.settings.stable_seconds)
        self.assertEqual(self.settings.gripper_force, 1.)

    async def test_lower_setting_reduces_target_rate(self):
        self.settings.gripper_close_speed = .002
        await self.service._grip()
        self.assert_slow_targets()
        self.assertGreaterEqual(self.events[-1][2]-10. + 1e-10, (.060-.0035)/.002)

    async def test_maximum_speed_reaches_target_at_30_mm_per_second(self):
        self.settings.gripper_close_speed = .030
        self.settings.validate()
        await self.service._grip()
        self.assert_slow_targets()
        # A 56.5 mm stroke at 30 mm/s takes 1.883 s, rounded to a 50 ms tick.
        travel_time = self.events[-1][2] - 10.
        self.assertGreaterEqual(travel_time + 1e-10, (.060-.0035)/.030)
        self.assertLessEqual(travel_time, (.060-.0035)/.030 + .05)
        self.assertAlmostEqual(.060-self.targets()[0], .0015)
        self.assertGreaterEqual(self.now-self.events[-1][2], self.settings.stable_seconds)

    async def test_maximum_speed_stall_keeps_feedback_lead_and_cancels_target(self):
        self.settings.gripper_close_speed = .030
        self.stuck = True
        with self.assertRaisesRegex(TimeoutError, '夹爪到位超时'):
            await self.service._grip()
        self.assertTrue(self.targets())
        for actual, target in zip(self.before_commands, self.targets()):
            self.assertLessEqual(actual-target, .002+1e-10)
        self.assertNotIn(self.settings.close_width, self.targets())
        self.assertEqual(self.events[-1][:2], ('hold_gripper', .060))

    async def test_maximum_speed_stop_cancels_pending_closure(self):
        self.settings.gripper_close_speed = .030
        self.on_command = lambda target: setattr(self.service, 'stop_requested', True)
        with self.assertRaises(Stopped):
            await self.service._grip()
        self.assertEqual(len(self.targets()), 1)
        self.assertEqual(self.events[-1][0], 'hold_gripper')

    async def test_scheduler_delay_never_causes_a_large_catchup_jump(self):
        # One late wakeup must not compress its missed trajectory into a jump.
        slept = False
        async def delayed_sleep(seconds):
            nonlocal slept
            self.now += seconds + (.8 if not slept else 0.)
            slept = True
        with patch('robot.service.asyncio', SimpleNamespace(sleep=delayed_sleep)):
            await self.service._grip()
        self.assert_slow_targets()

    async def test_close_from_within_arrival_tolerance_still_sends_final_target(self):
        self.width = self.settings.close_width + .0005
        await self.service._grip()
        self.assert_slow_targets(initial=.004)

    async def test_already_at_target_has_no_closing_ramp(self):
        self.width = self.settings.close_width
        await self.service._grip()
        self.assertEqual(self.targets(), [.0035])

    async def test_early_contact_cannot_shorten_closure(self):
        def telemetry():
            if self.targets():
                self.service.observe(SimpleNamespace(status='recognized',
                    contact=ContactResult(active=True, calibrated=True, valid=True)), self.now, 7)
        self.on_snapshot = telemetry
        await self.service._grip()
        self.assert_slow_targets()
        self.assertEqual(self.width, .0035)
        self.assertGreaterEqual(self.now - self.events[-1][2], self.settings.stable_seconds)
        self.assertIsNone(self.service.observation)

    async def test_cached_tactile_error_does_not_gate_position_command(self):
        self.service.observe(SimpleNamespace(status='error'), self.now-10, 99)
        await self.service._grip()
        self.assert_slow_targets()

    async def test_arrival_accepts_small_following_error(self):
        self.on_command = lambda target: setattr(self, 'width', target + .0004)
        await self.service._grip()
        self.assert_slow_targets()
        self.assertAlmostEqual(self.width, .0039)

    async def test_arrival_outside_one_mm_never_recognized_as_complete(self):
        self.on_command = lambda target: setattr(self, 'width', target + .0011)
        with self.assertRaisesRegex(TimeoutError, '夹爪到位超时') as caught:
            await self.service._grip()
        self.assertIn('当前 4.600 mm', str(caught.exception))
        self.assertIn('目标 3.500 mm', str(caught.exception))
        self.assert_slow_targets()
        self.assertEqual(self.events[-1][0], 'hold_gripper')

    async def test_brief_arrival_resets_continuous_stability_timer(self):
        reached_at = None
        def telemetry():
            nonlocal reached_at
            if self.targets() and self.targets()[-1] == self.settings.close_width:
                reached_at = self.now if reached_at is None else reached_at
                self.width = .006 if .09 < self.now-reached_at < .19 else .0035
        self.on_snapshot = telemetry
        await self.service._grip()
        self.assertGreaterEqual(self.now-reached_at, .19 + self.settings.stable_seconds)
        self.assert_slow_targets()

    async def test_stall_times_out_without_extra_commands_or_force(self):
        self.stuck = True
        with self.assertRaisesRegex(TimeoutError, '夹爪到位超时'):
            await self.service._grip()
        self.assertGreater(len(self.targets()), 0)
        self.assertGreaterEqual(min(self.targets()), .060-.002-1e-10)
        self.assertNotIn(self.settings.close_width, self.targets())
        self.assertEqual(self.events[-1][:2], ('hold_gripper', .060))
        self.assertEqual(self.settings.gripper_force, 1.)

    async def test_recovery_from_stall_has_no_large_jump(self):
        self.stuck = True
        def telemetry():
            if self.now >= 10.8 and self.targets():
                self.width = self.targets()[-1]
        self.on_snapshot = telemetry
        await self.service._grip()
        self.assert_slow_targets()
        for actual, target in zip(self.before_commands, self.targets()):
            self.assertLessEqual(actual-target, .002+1e-10)

    async def test_opening_overshoot_does_not_command_outside_configured_range(self):
        self.width = .0615  # Preparation accepts a small opening following error.
        await self.service._grip()
        self.assert_slow_targets()
        self.assertTrue(all(width <= self.settings.open_width for width in self.targets()))
        self.assertLessEqual(self.before_commands[0]-self.targets()[0], .002)

    async def test_invalid_initial_width_never_commands_motion(self):
        for width in (float('nan'), float('inf'), -.001, .101):
            with self.subTest(width=width):
                self.width = width
                with self.assertRaisesRegex(RuntimeError, '开口反馈无效'):
                    await self.service._grip()
                self.assertEqual(self.events, [])

    async def test_stale_or_invalid_initial_timestamp_never_commands_motion(self):
        for offset in (-1., 1., float('nan')):
            with self.subTest(offset=offset):
                self.timestamp_offset = offset
                with self.assertRaisesRegex(RuntimeError, '开口反馈无效'):
                    await self.service._grip()
                self.assertEqual(self.events, [])

    async def test_teaching_never_commands_closure(self):
        self.teaching = True
        with self.assertRaisesRegex(RuntimeError, '示教中不能闭合'):
            await self.service._grip()
        self.assertEqual(self.events, [])

    async def test_lost_feedback_after_command_cancels_pending_target(self):
        self.on_command = lambda target: setattr(self, 'timestamp_offset', -1.)
        with self.assertRaisesRegex(RuntimeError, '反馈无效或过期'):
            await self.service._grip()
        self.assertEqual(len(self.targets()), 1)
        self.assertGreater(self.targets()[0], self.settings.close_width)
        self.assertEqual(self.events[-1][0], 'hold_gripper')

    async def test_stop_before_command_never_closes(self):
        self.service.stop_requested = True
        with self.assertRaises(Stopped):
            await self.service._grip()
        self.assertEqual(self.events, [])

    async def test_stop_while_reading_initial_snapshot_never_closes(self):
        self.on_snapshot = lambda: setattr(self.service, 'stop_requested', True)
        with self.assertRaises(Stopped):
            await self.service._grip()
        self.assertEqual(self.events, [])

    async def test_stop_after_command_cancels_pending_target(self):
        self.stuck = True
        self.on_command = lambda target: setattr(self.service, 'stop_requested', True)
        with self.assertRaises(Stopped):
            await self.service._grip()
        self.assertEqual(len(self.targets()), 1)
        self.assertGreater(self.targets()[0], self.settings.close_width)
        self.assertEqual(self.events[-1][:2], ('hold_gripper', .060))

    async def test_stream_health_failure_after_command_cancels_target(self):
        self.on_command = lambda target: setattr(self.service, 'ready_check', lambda: False)
        with self.assertRaisesRegex(RuntimeError, '传感器已断流'):
            await self.service._grip()
        self.assertEqual(self.events[-1][0], 'hold_gripper')

    async def test_sensor_not_ready_never_closes(self):
        self.service.ready_check = lambda: False
        with self.assertRaisesRegex(RuntimeError, '传感器已断流'):
            await self.service._grip()
        self.assertEqual(self.events, [])

    async def test_hold_failure_reports_unconfirmed_motion(self):
        self.stuck = True
        async def io(name, *args):
            if name == 'hold_gripper':
                raise RuntimeError('CAN failure')
            return await self.io(name, *args)
        self.service.call.side_effect = io
        with self.assertRaisesRegex(RuntimeError, '无法确认夹爪保持'):
            await self.service._grip()

    async def test_cycle_cannot_recognize_or_return_before_arrival(self):
        self.stuck = True
        self.service._recognize = AsyncMock()
        self.service._release = AsyncMock()
        with self.assertLogs('robot.service', level='ERROR'):
            await self.service._cycle({'detect':[.1]*7, 'home':[0.]*7}, 7, 'rat', None)
        self.assertEqual(self.service.phase, 'fault')
        self.assertIn('夹爪到位超时', self.service.message)
        self.service._recognize.assert_not_called()
        self.service._release.assert_not_called()
        self.assertEqual([arg for name, arg, _ in self.events if name=='move'], [(.1,)*7])
        self.assertIn('hold_gripper', [name for name, _, _ in self.events])

    async def test_cycle_no_contact_at_target_never_tightens_again(self):
        self.settings.recognize_timeout = .3
        def feed_no_contact():
            self.service.observe(SimpleNamespace(status='no_contact', prediction=None,
                contact=ContactResult(calibrated=True, valid=True)), self.now, 7)
        self.on_snapshot = feed_no_contact
        report = Mock()
        await self.service._cycle({'detect':[.1]*7, 'home':[0.]*7}, 7, 'rat', report)
        self.assertEqual(self.service.phase, 'complete')
        self.assertEqual(self.targets()[0], .060)
        self.assertEqual(self.targets()[-2:], [.0035, .060])
        self.assertEqual(self.targets().count(.0035), 1)
        self.assertTrue(all(a>b for a,b in zip(self.targets()[:-2], self.targets()[1:-1])))
        self.assertEqual(self.events[-1][:2], ('move', (0.,)*7))
        report.assert_called_once_with(None, None, False)

    async def test_recognition_requires_frames_captured_after_arrival(self):
        await self.service._grip()
        started = self.now
        def feed():
            if self.now - started < .3:
                # Late result captured while closing: ignore even its old
                # stream/status, rather than failing or accepting it.
                self.service.observe(SimpleNamespace(status='error'), started-.1, 99)
            else:
                self.service.observe(SimpleNamespace(status='recognized',
                    prediction=SimpleNamespace(key='rat', accepted=True, confidence=.99),
                    contact=ContactResult(active=True, stable=True, calibrated=True, valid=True)),
                    self.now, 7)
        self.on_snapshot = feed
        key, _ = await self.service._recognize(7, 'rat')
        self.assertEqual(key, 'rat')
        # Strict mode already confirmed multiple fresh frames in the engine;
        # do not add the former 0.8-second hold after it has done that work.
        self.assertGreaterEqual(self.now - started, .3)
        self.assertLess(self.now - started, .4)
        self.assertEqual(self.targets()[-1], .0035)
        self.assertEqual(self.targets().count(.0035), 1)

    @staticmethod
    def recognition_state(key='rat', *, status='preview', accepted=True, confidence=.99, **contact):
        return SimpleNamespace(status=status,
            prediction=SimpleNamespace(key=key, accepted=accepted, confidence=confidence),
            contact=ContactResult(**({'calibrated': True, 'valid': True, 'active': True,
                                      'stable': True} | contact)))

    async def test_preview_confirms_fresh_frames_without_extra_point_eight_seconds(self):
        started = self.now
        stamps = []
        def feed():
            stamps.append(self.now)
            self.service.observe(self.recognition_state(), self.now, 7)
        self.on_snapshot = feed
        key, state = await self.service._recognize(7, 'rat')
        self.assertEqual(key, 'rat')
        self.assertEqual(state.status, 'preview')
        self.assertGreaterEqual(len(set(stamps[1:])), 3)
        self.assertGreaterEqual(self.now-stamps[1], .15-1e-9)
        self.assertLess(self.now-started, .3)
        self.assertEqual(self.events, [])  # Recognition itself issues no motion.

    async def test_preview_duplicate_timestamp_cannot_confirm(self):
        self.settings.recognize_timeout = .4
        stamp = self.now
        self.on_snapshot = lambda: self.service.observe(
            self.recognition_state(), stamp+.01, 7)
        self.assertEqual(await self.service._recognize(7, 'rat'), (None, None))

    async def test_preview_two_distinct_frames_are_insufficient_even_after_duration(self):
        self.settings.recognize_timeout = .4
        started = self.now
        def feed():
            stamp = started+.01 if self.now-started < .2 else started+.21
            self.service.observe(self.recognition_state(), stamp, 7)
        self.on_snapshot = feed
        self.assertEqual(await self.service._recognize(7, 'rat'), (None, None))

    async def test_preview_rejection_or_class_change_restarts_confirmation(self):
        started = self.now
        def feed():
            tick = round((self.now-started)/.05)
            state = self.recognition_state('ox' if tick == 3 else 'rat', accepted=tick != 4)
            self.service.observe(state, self.now, 7)
        self.on_snapshot = feed
        key, _ = await self.service._recognize(7, 'rat')
        self.assertEqual(key, 'rat')
        self.assertAlmostEqual(self.now-started, .4, places=6)

    async def test_invalid_contact_cannot_use_strict_fast_path(self):
        for contact in ({'calibrated': False}, {'valid': False}, {'active': False}, {'stable': False}):
            with self.subTest(contact=contact):
                self.settings.recognize_timeout = .3
                state = self.recognition_state(status='recognized')
                from dataclasses import replace
                state.contact = replace(state.contact, **contact)
                self.on_snapshot = lambda: self.service.observe(state, self.now, 7)
                self.assertEqual(await self.service._recognize(7, 'rat'), (None, None))

    async def test_low_confidence_preview_never_becomes_a_final_result(self):
        for confidence in (.49, .49999, float('nan')):
            with self.subTest(confidence=confidence):
                self.settings.recognize_timeout = .4
                state = self.recognition_state(confidence=confidence)
                self.on_snapshot = lambda: self.service.observe(state, self.now, 7)
                self.assertEqual(await self.service._recognize(7, 'rat'), (None, None))
                self.assertIn('置信度不足', self.service.message)
                self.assertEqual(self.events, [])

    async def test_fifty_percent_preview_can_confirm(self):
        state = self.recognition_state(confidence=.5)
        self.on_snapshot = lambda: self.service.observe(state, self.now, 7)
        key, _ = await self.service._recognize(7, 'rat')
        self.assertEqual(key, 'rat')

    async def test_low_confidence_resets_previous_matching_frames(self):
        started = self.now
        def feed():
            tick = round((self.now-started)/.05)
            state = self.recognition_state(confidence=.49 if tick == 3 else .9)
            self.service.observe(state, self.now, 7)
        self.on_snapshot = feed
        key, _ = await self.service._recognize(7, 'rat')
        self.assertEqual(key, 'rat')
        self.assertAlmostEqual(self.now-started, .35, places=6)
