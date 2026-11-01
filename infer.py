"""Live sensor inference using CPU ONNX only; no training framework is imported."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
import cv2
from core.camera import CameraInfo, CameraStream
from core.config import load_config
from core.contact import make_difference
from core.state_machine import RecognitionEngine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='实时差分图生肖推理（仅ONNX Runtime，无PyTorch）')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--calibrate-empty', action='store_true', help='确认当前没有物体接触，启动后采集空载基准')
    parser.add_argument('--headless', action='store_true', help='不显示OpenCV窗口，输出JSON状态')
    parser.add_argument('--max-frames', type=int, default=0, help='处理指定数量的新帧后退出，0表示持续运行')
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error('--max-frames 不能为负数')
    config = load_config(args.config)
    if args.model:
        config = replace(config, recognition=replace(config.recognition, model_path=str(args.model)))
    engine = RecognitionEngine(config)
    print(engine.model_status, flush=True)
    if not args.calibrate_empty:
        parser.error('需要先移开物体，再使用 --calibrate-empty 明确确认空载；或运行 main.py 交互校准')
    stream = CameraStream(CameraInfo(args.camera, f'Sensor {args.camera}'), target_fps=config.sensor.capture_fps)
    stream.start()
    seen = (0, 0)
    count = 0
    frames = []
    calibration_serial = None
    calibrated_once = False
    previous_status = None
    started = time.monotonic()
    try:
        if not stream.wait_ready(5):
            raise RuntimeError(stream.error or '传感器打开失败')
        while True:
            packet = stream.frame_packet()
            if packet is None or time.monotonic()-packet.timestamp > config.recognition.max_frame_age:
                if time.monotonic()-started > 5:
                    raise RuntimeError('传感器信号中断，停止识别；请重连后重新空载校准')
                time.sleep(.01)
                continue
            started = time.monotonic()
            token = (packet.serial, packet.sequence)
            if token == seen:
                time.sleep(.01)
                continue
            seen = token
            # Allow auto exposure to settle before sampling the empty reference.
            if packet.sequence < 20:
                continue
            if not engine.detector.calibrated:
                if calibrated_once:
                    raise RuntimeError('空载基准已失效，请重新确认空载后启动')
                if calibration_serial is not None and packet.serial != calibration_serial:
                    raise RuntimeError('校准期间相机重连，请重新启动并确认空载')
                calibration_serial = packet.serial
                frames.append(packet.frame)
                if len(frames) < config.contact.calibration_frames:
                    continue
                engine.calibrate(frames)
                calibrated_once = True
                frames.clear()
                print('空载校准完成', flush=True)
                continue
            if packet.serial != calibration_serial:
                raise RuntimeError('传感器已重连，需要重新确认空载')
            result = engine.process(packet.frame, sequence=packet.sequence, timestamp=packet.timestamp, stream_id=packet.serial)
            count += 1
            if result.status != previous_status or result.new_record:
                print(json.dumps({'status': result.status, 'message': result.message,
                                  'zodiac': result.prediction.key if result.status=='recognized' else None,
                                  'contact_area': result.contact.area_fraction, 'new_record': result.new_record},
                                 ensure_ascii=False), flush=True)
                previous_status = result.status
            if not args.headless:
                diff = make_difference(packet.frame, engine.detector.baseline) if engine.detector.baseline is not None else packet.frame*0
                display = diff.copy()
                cv2.putText(display, result.status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 210, 210), 2)
                cv2.imshow('TouchSign - absolute difference (Q to quit)', display)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            if args.max_frames and count >= args.max_frames:
                break
    except (RuntimeError, ValueError) as exc:
        print(str(exc), flush=True)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
