"""Regression for physical presses queued while the collector is writing an image."""
import sys
import unittest


@unittest.skipUnless(sys.platform == 'win32', 'uses the Windows message queue')
class CollectorKeyTests(unittest.TestCase):
    def test_two_presses_survive_a_queued_auto_repeat(self):
        import ctypes
        from ctypes import wintypes
        from threading import get_native_id
        from collect import _ignore_key_repeats

        user32 = ctypes.WinDLL('user32', use_last_error=True)
        user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        user32.PeekMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                        wintypes.UINT, wintypes.UINT, wintypes.UINT)
        message = wintypes.MSG()
        # Use only thread messages (HWND -1). No camera, window or global input.
        user32.PeekMessageW(ctypes.byref(message), -1, 0, 0, 0)
        down, up, char = 0x0100, 0x0101, 0x0102
        with _ignore_key_repeats():
            for kind, flags in [(down, 1), (char, 1), (down, 0x40000001), (char, 0x40000001),
                                (up, 0xC0000001), (down, 1), (char, 1), (up, 0xC0000001)]:
                self.assertTrue(user32.PostThreadMessageW(get_native_id(), kind, ord(' '), flags))
            received = []
            while user32.PeekMessageW(ctypes.byref(message), -1, down, char, 1):
                received.append(message.message)
        self.assertEqual(received, [down, char, 0, 0, up, down, char, up])
