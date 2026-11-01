"""TouchSign keyboard collector in an OpenCV HighGUI window."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
from core.camera import CameraInfo, CameraManager
from core.collection import CollectionController
from core.config import load_config
from core.dataset import dataset_summary
from gui.collection_canvas import Canvas, LABEL_SHORTCUTS

WINDOW_NAME = 'TouchSign Capture'  # Some HighGUI backends interpret window names using a local code page.
LABEL_KEYS = {ord(key.lower()): label for label, key in LABEL_SHORTCUTS.items()}
LEFT = {0x250000, 0xFF51, 0x01000012, 63234}  # Win32, GTK, Qt, Cocoa
RIGHT = {0x270000, 0xFF53, 0x01000014, 63235}

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    _user32 = ctypes.windll.user32
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    _MessageHook = ctypes.WINFUNCTYPE(wintypes.LPARAM, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    _user32.SetWindowsHookExW.argtypes = (ctypes.c_int, _MessageHook, wintypes.HINSTANCE, wintypes.DWORD)
    _user32.SetWindowsHookExW.restype = wintypes.HHOOK
    _user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    _user32.CallNextHookEx.restype = wintypes.LPARAM
    _user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)


def _window_active() -> bool:
    if sys.platform != 'win32':
        return True
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(_user32.GetForegroundWindow(), ctypes.byref(pid))
    return pid.value == os.getpid()


@contextmanager
def _ignore_key_repeats():
    """Keep distinct presses even when several arrive during a slow PNG write."""
    if sys.platform != 'win32':
        yield
        return

    @_MessageHook
    def filter_repeat(code, removed, address):
        if code == 0 and removed == 1:  # HC_ACTION, PM_REMOVE
            message = ctypes.cast(address, ctypes.POINTER(wintypes.MSG)).contents
            # HighGUI returns just the key code. Before it reads WM_KEYDOWN or
            # WM_CHAR, drop messages marked 'already down' by Windows (bit 30).
            if message.message in (0x0100, 0x0102) and message.lParam & (1 << 30):
                message.message = 0  # WM_NULL
        return _user32.CallNextHookEx(None, code, removed, address)

    # WH_GETMESSAGE for this HighGUI thread only.
    hook = _user32.SetWindowsHookExW(3, filter_repeat, None, threading.get_native_id())
    if not hook:
        raise ctypes.WinError()
    try:
        yield
    finally:
        _user32.UnhookWindowsHookEx(hook)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OpenCV 快捷采集。移开物体后按 C 校准，选类后按空格拍摄。')
    parser.add_argument('--config', help='传感器与接触检测配置文件')
    parser.add_argument('--camera', type=int, default=0, help='传感器设备编号（默认 0）')
    parser.add_argument('--output', type=Path, default=Path('data/raw'), help='保存样本的目录')
    parser.add_argument('--font', type=Path, help='指定支持简体中文的 TTF / TTC / OTF 字体')
    parser.add_argument('--target', type=int, default=100, help='每轮连拍目标张数')
    parser.add_argument('--interval', type=float, default=0.5, help='连拍的最短间隔，单位为秒')
    parser.add_argument('--per-press', type=int, default=20, help='连拍时单次接触最多保存的张数')
    parser.add_argument('--notes', default='', help='空载校准时写入会话的备注')
    parser.add_argument('--summary', action='store_true', help='只输出已有数据统计')
    args = parser.parse_args(argv)
    if args.camera < 0 or args.target < 1 or args.per_press < 1 or not 0.1 <= args.interval <= 60:
        parser.error('设备编号须为非负整数，目标及每次接触张数须为正整数，间隔范围为 0.1 到 60 秒')
    return args


def run_window(controller: CollectionController, args: argparse.Namespace, canvas: Canvas) -> None:
    labels = tuple(LABEL_SHORTCUTS)
    show_diff = False
    cancel = threading.Event()
    calibration = None
    controller.message = '移开所有物体后按 C 确认空载并校准。'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, *canvas.size)
    with _ignore_key_repeats(), ThreadPoolExecutor(max_workers=1) as worker:
        try:
            while True:
                # The median reference can take a moment at full sensor resolution.
                # Pump HighGUI on this thread even while the worker is calibrating.
                if calibration is not None and calibration.done():
                    try:
                        calibration.result()
                        show_diff = True
                    except (ValueError, RuntimeError, OSError) as exc:
                        controller.message = str(exc)
                    calibration = None
                if calibration is None:
                    if controller.recording and not _window_active():
                        controller.stop()
                    raw, difference = controller.poll()
                    calibrated = controller.detector.calibrated
                    show_diff = show_diff and calibrated
                    preview = difference if show_diff else raw
                    contact = controller.contact.reason if controller.contact is not None else '等待空载校准'
                    if raw is None:
                        contact = '等待传感器图像'
                else:
                    packet = controller.camera.frame_packet()
                    preview = packet.frame if packet is not None and time.monotonic() - packet.timestamp <= .5 else None
                    calibrated, show_diff = False, False
                    contact = '正在采集空载基准，请保持静止'
                counts = controller.writer.counts
                frame = canvas.render(preview, label=controller.label,
                                      total=sum(counts.values()), class_count=counts[controller.label],
                                      contact=contact, message=controller.message, diff=show_diff,
                                      recording=controller.recording, calibrated=calibrated, camera_index=args.camera)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKeyEx(20)
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:  # Win32 removes the window handle when the user clicks X.
                    break
                while key != -1:
                    if ord('A') <= key <= ord('Z'):
                        key += 32
                    if key == ord('q'):
                        return
                    if calibration is None:
                        if key == ord('c'):
                            controller.stop()
                            controller.message = '正在空载校准，请保持传感器静止。'
                            calibration = worker.submit(controller.calibrate, confirmed_unloaded=True,
                                                        notes=args.notes, cancel=cancel)
                        elif key in LABEL_KEYS:
                            controller.choose_label(LABEL_KEYS[key])
                        elif key in LEFT or key in RIGHT:
                            step = 1 if key in RIGHT else -1
                            base = labels.index(controller.label) if controller.label in labels else (-1 if step == 1 else 0)
                            controller.choose_label(labels[(base + step) % len(labels)])
                        elif key == ord(' '):
                            controller.capture_once(expected_label=controller.label)
                        elif key == ord('r'):
                            if controller.recording:
                                controller.stop()
                            else:
                                try:
                                    controller.start(confirmed=controller.label is not None, target=args.target,
                                                     interval=args.interval, max_per_episode=args.per_press)
                                except ValueError as exc:
                                    controller.message = str(exc)
                        elif key == 27 and controller.recording:
                            controller.stop()
                        elif key == ord('v'):
                            if controller.detector.calibrated:
                                show_diff = not show_diff
                            else:
                                controller.message = '先移开物体并按 C 校准，即可查看差分图。'
                    key = cv2.pollKey()
        finally:
            cancel.set()
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass  # Already closed from the title bar.
            controller.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary:
        print(json.dumps(dataset_summary(args.output), ensure_ascii=False, indent=2))
        return 0
    manager = CameraManager()
    try:
        canvas = Canvas(args.font)
        config = load_config(args.config)
        controller = CollectionController(manager, args.output, contact_config=config.contact)
        stream = manager.select(CameraInfo(args.camera, ''), target_fps=config.sensor.capture_fps, encode_jpeg=False)
        if not stream.wait_ready(5.0):
            raise RuntimeError(f'无法打开相机 {args.camera}。请检查设备编号和相机占用情况。')
        run_window(controller, args, canvas)
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f'采集失败：{exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
    return 0


if __name__ == '__main__':
    # Redirected Windows streams otherwise use the legacy system code page.
    for output in (sys.stdout, sys.stderr):
        if output is not None and not output.isatty():
            output.reconfigure(encoding='utf-8')
    raise SystemExit(main())
