"""Typed runtime configuration; deployment data is separate from bundled resources."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
import math
import os
import sys
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN = bool(getattr(sys, 'frozen', False))
DATA_ROOT = (Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'TouchSign') if FROZEN else PROJECT_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'


def resolve_path(value: str | Path, *, writable: bool = False) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if FROZEN and not writable:
        external = Path(sys.executable).parent / path
        if external.exists():
            return external
    return (DATA_ROOT if writable else PROJECT_ROOT) / path


@dataclass(frozen=True, slots=True)
class AppConfig:
    title: str = '触象 · TouchSign | 视触觉生肖识别系统'
    host: str = '127.0.0.1'
    port: int = 8080
    show: bool = True
    reload: bool = False
    native: bool = False
    window_width: int = 1600
    window_height: int = 1000
    favicon: str = 'assets/brand/favicon.svg'


@dataclass(frozen=True, slots=True)
class SensorConfig:
    exposure_ms: float = 8.0
    gain: float = 1.2
    gel_thickness_mm: float = 0.1
    signal_fps: int = 60
    scan_max_index: int = 6
    jpeg_quality: int = 82
    capture_fps: float = 20.0


@dataclass(frozen=True, slots=True)
class ContactConfig:
    calibration_frames: int = 30
    calibration_max_motion: float = 3.0
    pixel_threshold: float = 12.0
    noise_multiplier: float = 6.0
    min_area: float = 0.012
    release_area: float = 0.006
    max_area: float = 0.70
    min_score: float = 0.002
    stable_frames: int = 3
    stable_seconds: float = 0.20
    max_frame_motion: float = 0.025
    border_fraction: float = 0.04

    def __post_init__(self) -> None:
        if not (0 <= self.release_area < self.min_area < self.max_area <= 1):
            raise ValueError('contact 面积阈值必须满足 0 <= release_area < min_area < max_area <= 1')
        if self.calibration_frames < 10 or self.stable_frames < 1:
            raise ValueError('空载校准至少10帧，稳定帧数至少1帧')
        if not (0 < self.pixel_threshold < 255 and self.noise_multiplier > 0):
            raise ValueError('差分像素阈值或噪声系数无效')
        if not (0 <= self.border_fraction < .4 and 0 <= self.stable_seconds < 60):
            raise ValueError('接触边缘或稳定时间配置无效')
        if not (0 < self.min_score < 1 and 0 < self.max_frame_motion < 1 and self.calibration_max_motion > 0):
            raise ValueError('接触变化量配置无效')


@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    confidence_threshold: float = 0.85
    history_limit: int = 4
    model_path: str = 'models/best.onnx'
    manifest_path: str = ''
    confirmation_frames: int = 4
    max_frame_age: float = 0.5
    log_path: str = 'logs/recognition.csv'
    demo_image: str = 'assets/demo/tactile-rat.png'
    demo_auto_cycle_seconds: float = 0.0
    initial_zodiac: str = 'rat'
    preview: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError('confidence_threshold 必须在 (0, 1]')
        if self.confirmation_frames < 1 or not 0 < self.max_frame_age <= 5 or self.history_limit < 1:
            raise ValueError('识别确认帧数、帧龄或记录数量配置无效')


@dataclass(frozen=True, slots=True)
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)

    @property
    def demo_image_path(self) -> Path:
        return resolve_path(self.recognition.demo_image)

    @property
    def favicon_path(self) -> Path | None:
        return resolve_path(self.app.favicon) if self.app.favicon else None


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _build(cls, raw: dict):
    defaults = cls()
    values = {}
    for item in fields(defaults):
        default = getattr(defaults, item.name)
        value = raw.get(item.name, default)
        try:
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError('must be YAML true/false')
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError('must be finite')
            else:
                value = str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{cls.__name__}.{item.name}: {exc}') from exc
        values[item.name] = value
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _mapping(yaml.safe_load(config_path.read_text(encoding='utf-8'))) if config_path.is_file() else {}
    sensor = _mapping(raw.get('sensor')).copy()
    sensor.update(_mapping(sensor.pop('camera', {})))
    config = Config(
        app=_build(AppConfig, _mapping(raw.get('app'))),
        sensor=_build(SensorConfig, sensor),
        recognition=_build(RecognitionConfig, _mapping(raw.get('recognition'))),
        contact=_build(ContactConfig, _mapping(raw.get('contact'))),
    )
    if not 1 <= config.app.port <= 65535 or config.sensor.capture_fps <= 0:
        raise ValueError('端口或摄像头帧率无效')
    return config
