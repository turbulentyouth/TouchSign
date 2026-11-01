"""Empty-surface calibration and fail-closed optical contact gating.

Calibration must be explicitly confirmed as unloaded by the operator. Optical
change is neither calibrated physical force nor proof of object identity.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Sequence
import cv2
import numpy as np
from core.config import ContactConfig

INPUT_KIND = 'absdiff_bgr_v1'


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('图像必须是 uint8 BGR 三通道图像')
    if min(frame.shape[:2]) < 32:
        raise ValueError('图像尺寸太小')


def make_difference(frame: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Canonical model input: unsigned absolute BGR difference without scaling."""
    validate_frame(frame)
    validate_frame(baseline)
    if frame.shape != baseline.shape:
        raise ValueError('当前图像与空载基准尺寸不同，请重新校准')
    return cv2.absdiff(frame, baseline)


@dataclass(frozen=True, slots=True)
class ContactResult:
    calibrated: bool = False
    active: bool = False
    stable: bool = False
    score: float = 0.0
    area_fraction: float = 0.0
    duration_s: float = 0.0
    stability: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    reason: str = '请先确认空载并校准'
    valid: bool = True


class ContactDetector:
    def __init__(self, config: ContactConfig | None = None) -> None:
        self.config = config or ContactConfig()
        self.reset()

    @property
    def baseline(self) -> np.ndarray | None:
        return self._baseline

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    def reset(self) -> None:
        self._baseline: np.ndarray | None = None
        self._threshold: np.ndarray | None = None
        self.clear_temporal()

    def clear_temporal(self) -> None:
        self._active = False
        self._since: float | None = None
        self._last_timestamp: float | None = None
        self._stable_since: float | None = None
        self._stable_count = 0
        self._previous: np.ndarray | None = None

    def calibrate(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        self.reset()
        if len(frames) < 10:
            raise ValueError('空载校准需要至少10帧新图像')
        for frame in frames:
            validate_frame(frame)
        if any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError('校准期间图像尺寸发生变化')
        stack = np.stack(frames)
        baseline = np.median(stack, axis=0).astype(np.uint8)
        errors = np.abs(stack.astype(np.float32) - baseline.astype(np.float32))
        if float(np.max(np.mean(errors, axis=(1, 2, 3)))) > self.config.calibration_max_motion:
            raise ValueError('校准期间画面不稳定，请移开物体并保持传感器静止')
        noise = np.median(errors, axis=0).max(axis=2)
        self._threshold = np.maximum(self.config.pixel_threshold, noise * self.config.noise_multiplier + 3).astype(np.float32)
        baseline.setflags(write=False)
        self._baseline = baseline
        return baseline.copy()

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> ContactResult:
        if not self.calibrated:
            return ContactResult()
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if self._last_timestamp is not None and stamp <= self._last_timestamp:
            self.clear_temporal()
            return ContactResult(calibrated=True, valid=False, reason='等待新的传感器帧')
        self._last_timestamp = stamp
        try:
            diff = make_difference(frame, self._baseline)
        except ValueError as exc:
            self.reset()
            return ContactResult(reason=str(exc))
        smooth = cv2.GaussianBlur(diff, (5, 5), 0).max(axis=2).astype(np.float32)
        height, width = smooth.shape
        by = int(height * self.config.border_fraction)
        bx = int(width * self.config.border_fraction)
        roi = np.zeros(smooth.shape, dtype=bool)
        roi[by:height-by, bx:width-bx] = True
        changed = (smooth > self._threshold) & roi
        mask = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        area = float(mask.sum()) / max(int(roi.sum()), 1)
        score = float(smooth[roi].mean()) / 255.0
        threshold = self.config.release_area if self._active else self.config.min_area
        if area > self.config.max_area:
            self.clear_temporal()
            return ContactResult(calibrated=True, valid=False, score=score, area_fraction=area,
                                 reason='全局画面变化过大：检查照明、相机位置或重新空载校准')
        if area < threshold or score < self.config.min_score:
            self.clear_temporal()
            return ContactResult(calibrated=True, score=score, area_fraction=area, reason='未接触')
        if not self._active:
            self._since = stamp
        self._active = True
        motion = 1.0 if self._previous is None else float(np.mean(np.abs(smooth-self._previous))) / 255.0
        self._previous = smooth
        if motion <= self.config.max_frame_motion:
            self._stable_count += 1
            if self._stable_since is None:
                self._stable_since = stamp
        else:
            self._stable_count = 0
            self._stable_since = None
        stable = (self._stable_count >= self.config.stable_frames and self._stable_since is not None
                  and stamp-self._stable_since >= self.config.stable_seconds)
        ys, xs = np.nonzero(mask)
        return ContactResult(
            calibrated=True, active=True, stable=stable, score=score, area_fraction=area,
            duration_s=max(0, stamp-self._since), stability=min(1, self._stable_count/self.config.stable_frames),
            center_x=float(xs.mean()) if xs.size else 0, center_y=float(ys.mean()) if ys.size else 0,
            reason='接触稳定' if stable else '接触中，请保持稳定',
        )
