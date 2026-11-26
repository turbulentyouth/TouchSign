"""Quick manual capture: select label, press the sensor, hit Space."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys

import cv2

from core.camera import CameraInfo, CameraManager, describe_cameras, preferred_camera
from core.config import PROJECT_ROOT, load_config
from core.dataset import DATASET_LABELS
from core.instance import desktop_instance
from core.manual_collection import ManualCollector, ManualDatasetWriter
from core.paths import user_data_root
from gui.collection_canvas import LABEL_SHORTCUTS
from gui.manual_collection_canvas import ManualCanvas

WINDOW_NAME = 'TouchSign Manual Capture'
LABEL_KEYS = {ord(shortcut.lower()): label for label, shortcut in LABEL_SHORTCUTS.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='手工快速采集：选标签、手工按压、空格保存 180° 图像。')
    profile = 'windows' if sys.platform == 'win32' else 'ubuntu'
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / f'config/{profile}/config.yaml')
    parser.add_argument('--camera', type=int, help='指定相机编号；默认自动查找配置中的 VBTS 传感器')
    parser.add_argument('--label', choices=DATASET_LABELS, help='启动时选好标签，也可在窗口中按键选类')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/manual', help='保存目录')
    parser.add_argument('--interval', type=float, default=.2, help='连拍最短间隔，默认 0.2 秒')
    parser.add_argument('--font', type=Path, help='中文 TTF / TTC / OTF 字体路径')
    parser.add_argument('--list-cameras', action='store_true', help='列出相机后退出')
    args = parser.parse_args(argv)
    if (args.camera is not None and args.camera < 0) or not .1 <= args.interval <= 60:
        parser.error('相机编号须为非负整数，连拍间隔须为 0.1–60 秒')
    return args


def run_window(camera: CameraManager, collector: ManualCollector, canvas: ManualCanvas, index: int):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, *canvas.size)
    try:
        while True:
            packet = camera.frame_packet()
            preview = collector.poll(packet)
            cv2.imshow(WINDOW_NAME, canvas.render_manual(
                preview, label=collector.label, total=sum(collector.counts.values()),
                class_count=collector.counts[collector.label], recording=collector.recording,
                pending=collector.pending, message=collector.error or collector.message,
                camera_index=index,
            ))
            key = cv2.waitKeyEx(15)
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if ord('A') <= key <= ord('Z'):
                key += 32
            if key == ord('q'):
                break
            if key in LABEL_KEYS:
                collector.choose_label(LABEL_KEYS[key])
            elif key == ord(' '):
                collector.capture_once(packet)
            elif key == ord('r'):
                collector.start(packet)
            elif key in (ord('z'), 8, 0xff08):  # Backspace: Windows / Linux Qt
                collector.undo()
            elif key == 27:
                collector.stop()
    finally:
        cv2.destroyAllWindows()


def choose_windows_camera(camera, cameras, canvas, fps):
    """Confirm the sensor visually instead of silently using a laptop webcam."""
    available = [item for item in cameras if item.available]
    if not available:
        raise RuntimeError('未找到摄像头，请插好 USB 传感器、关闭占用相机的软件后重试。')
    index = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, *canvas.size)
    try:
        while True:
            selected = available[index]
            stream = camera.select(selected, target_fps=fps, encode_jpeg=False)
            ready = stream.wait_ready()
            while True:
                packet = camera.frame_packet()
                frame = packet.frame if packet is not None else None
                cv2.imshow(WINDOW_NAME, canvas.render_manual(
                    frame, label=None, total=0, class_count=0, recording=False, pending=0,
                    message=('确认这是触觉传感器：Enter 使用，Tab 换相机，Q 退出。' if ready else
                             '相机打开失败：Tab 换相机，Q 退出。') + f'  {index + 1}/{len(available)}',
                    camera_index=selected.index))
                key = cv2.waitKeyEx(30)
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    return None
                if key in (10, 13) and ready:
                    return selected
                if key == 9:
                    index = (index + 1) % len(available)
                    break
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)
    if not args.list_cameras and sys.platform == 'linux' and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    ):
        print('请在 Ubuntu 桌面终端运行此命令，采集窗口需要图形显示环境。', file=sys.stderr)
        return 1
    config = load_config(args.config)
    with desktop_instance(user_data_root() / 'desktop.lock', '') as existing:
        if existing is not None:
            print('TouchSign 已在运行，请先退出原程序以释放传感器。', file=sys.stderr)
            return 1
        camera = CameraManager()
        collector = None
        try:
            if args.list_cameras:
                print(describe_cameras(camera.scan(max_index=config.sensor.scan_max_index)))
                return 0
            canvas = ManualCanvas(args.font)
            if args.camera is None:
                cameras = camera.scan(max_index=config.sensor.scan_max_index,
                                      name_filter=config.sensor.preferred_camera_name)
                if sys.platform == 'win32':
                    selected = choose_windows_camera(camera, cameras, canvas, config.sensor.capture_fps)
                    if selected is None:
                        return 0
                else:
                    selected = preferred_camera(cameras, config.sensor.preferred_camera_name)
                if selected is None:
                    raise RuntimeError('未找到可用传感器。使用 --list-cameras 查看设备，或 --camera 编号 指定。\n'
                                       + describe_cameras(cameras))
            else:
                selected = CameraInfo(args.camera, f'传感器 #{args.camera}')
            stream = camera.select(selected, target_fps=config.sensor.capture_fps, encode_jpeg=False)
            if not stream.wait_ready():
                raise RuntimeError(stream.error or '打开传感器超时')
            writer = ManualDatasetWriter(args.output, camera=asdict(selected))
            collector = ManualCollector(writer, label=args.label, interval=args.interval)
            print(f'保存目录：{writer.session_dir}', flush=True)
            print('选标签 → 手工按压 → 空格保存。Z / 退格撤回最近一张；R 连拍，Esc 停止，Q 退出。', flush=True)
            try:
                run_window(camera, collector, canvas, selected.index)
            except KeyboardInterrupt:
                pass
        finally:
            camera.close()
            if collector is not None:
                collector.close()
                print(f'已保存 {sum(collector.counts.values())} 张：{dict(collector.counts)}', flush=True)
                if collector.error:
                    print(collector.error, file=sys.stderr)
        return 1 if collector is not None and collector.error else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f'手工采集失败：{exc}', file=sys.stderr)
        raise SystemExit(1)
