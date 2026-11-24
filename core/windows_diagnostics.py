"""Read-only Windows process diagnostics; never changes token or policy."""
import ctypes
from ctypes import wintypes
import sys


def token_integrity():
    if sys.platform != 'win32':
        return 'not-windows'
    advapi = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = wintypes.HANDLE()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(handle)):
        return 'unavailable'
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(handle, 25, None, 0, ctypes.byref(size))
        data = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(handle, 25, data, size, ctypes.byref(size)):
            return 'unavailable'
        sid = ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p))[0]
        advapi.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        advapi.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        advapi.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
        count = advapi.GetSidSubAuthorityCount(sid)[0]
        rid = advapi.GetSidSubAuthority(sid, count - 1)[0]
        return {'rid': rid, 'level': 'low' if rid < 8192 else 'medium' if rid < 12288 else 'high'}
    finally:
        kernel.CloseHandle(handle)
