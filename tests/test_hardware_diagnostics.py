from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.check_hardware import main


class HardwareDiagnosticsTests(unittest.TestCase):
    def run_diagnostic(self, messages, link=None):
        bus = Mock()
        bus.recv.side_effect = messages
        output = io.StringIO()
        clock = [0.] + [.1 * (i + 1) for i in range(len(messages))] + [3.]
        with patch('tools.check_hardware.inventory', return_value={
                'can_commands_sent': False, 'errors': []}), \
                patch('tools.check_hardware.check_socketcan'), \
                patch('tools.check_hardware.can_link', **({'side_effect': link} if isinstance(link, Exception)
                      else {'return_value': link or {'ifname': 'can_follower'}})), \
                patch('tools.check_hardware.time.monotonic', side_effect=clock), \
                patch('can.Bus') as factory, redirect_stdout(output):
            factory.return_value.__enter__.return_value = bus
            code = main(['--can-channel', 'can_follower'])
        bus.send.assert_not_called()
        return code, json.loads(output.getvalue())

    def test_error_frames_are_not_mistaken_for_arm_feedback(self):
        message = SimpleNamespace(is_error_frame=True, arbitration_id=0x24)
        code, report = self.run_diagnostic([message, message])
        self.assertEqual(code, 1)
        self.assertEqual(report['can_receive']['frames'], 0)
        self.assertEqual(report['can_receive']['error_frames'], {'0x24': 2})
        self.assertEqual(len(report['errors']), 2)
        self.assertIn('can_link_before', report)
        self.assertIn('can_link_after', report)

    def test_data_alongside_errors_still_reports_bus_problem(self):
        code, report = self.run_diagnostic([
            SimpleNamespace(is_error_frame=False, arbitration_id=0x2A1),
            SimpleNamespace(is_error_frame=True, arbitration_id=0x24)])
        self.assertEqual(code, 1)
        self.assertEqual(report['can_receive']['frames'], 1)
        self.assertEqual(report['can_receive']['ids'], ['0x2a1'])
        self.assertEqual(len(report['errors']), 1)

    def test_receive_still_works_without_ip_command(self):
        code, report = self.run_diagnostic([
            SimpleNamespace(is_error_frame=False, arbitration_id=0x2A1)],
            FileNotFoundError('ip unavailable'))
        self.assertEqual(code, 0)
        self.assertEqual(report['can_receive']['frames'], 1)
        self.assertIn('ip unavailable', report['can_link_warning'])
        self.assertFalse(report['can_commands_sent'])
