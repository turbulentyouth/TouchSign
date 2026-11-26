"""Desktop ownership on Linux and Windows; released by the OS after a crash."""
from contextlib import contextmanager
import errno
import json
import os
import sys


def _lock(handle, unlock=False):
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        # Lock beyond the metadata so other processes can still read the URL.
        # Use the Win32 error code directly: CRT locking can lose the errno.
        class Overlapped(ctypes.Structure):
            _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                        ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
                        ('hEvent', wintypes.HANDLE)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        operation = kernel.UnlockFileEx if unlock else kernel.LockFileEx
        operation.argtypes = ([wintypes.HANDLE, wintypes.DWORD] +
                              ([] if unlock else [wintypes.DWORD]) +
                              [wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)])
        operation.restype = wintypes.BOOL
        overlap = Overlapped(Offset=65536)
        args = [msvcrt.get_osfhandle(handle.fileno()), 0 if unlock else 3]
        if not unlock:
            args.append(0)
        if not operation(*args, 1, 0, ctypes.byref(overlap)):
            error = ctypes.get_last_error()
            if not unlock and error == 33:  # ERROR_LOCK_VIOLATION
                raise BlockingIOError(errno.EAGAIN, 'TouchSign already running')
            raise ctypes.WinError(error)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def desktop_instance(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink this file: another process may already be waiting on its inode.
    with path.open('a+', encoding='utf-8') as handle:
        try:
            _lock(handle)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            handle.seek(0)
            try:
                existing = json.load(handle)
            except (ValueError, OSError):
                existing = {}
            yield existing if isinstance(existing, dict) else {}
            return
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({'pid': os.getpid(), 'url': url}, handle)
            handle.flush()
            yield None
        finally:
            _lock(handle, unlock=True)
