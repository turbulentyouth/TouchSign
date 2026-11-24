"""Robot settings and taught points share the application's user config.yaml."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
import os
import tempfile
import sys
import yaml
from core.paths import user_data_root


def joints(value):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError('需要七个关节角（rad）')
    result = tuple(float(x) for x in value)
    if any(not math.isfinite(x) or abs(x) > 2 * math.pi for x in result):
        raise ValueError('关节角必须为有限值，且在 ±2π 内；真机还会校验自身限位')
    return result


@dataclass
class RobotSettings:
    backend: str = 'mock'
    channel: str = field(default_factory=lambda: 'can0' if sys.platform == 'linux' else '0')
    speed_percent: int = 10
    open_width: float = 0.06
    close_width: float = 0.02
    gripper_force: float = 1.0
    gripper_close_speed: float = 0.005  # m/s, software position ramp
    grip_timeout: float = 30.0
    move_timeout: float = 20.0
    recognize_timeout: float = 8.0
    release_timeout: float = 5.0
    joint_tolerance: float = 0.02
    stable_seconds: float = 0.35
    points: dict = field(default_factory=dict)

    def validate(self):
        if self.backend not in ('mock', 'nero'):
            raise ValueError('未知机械臂后端')
        for name in ('open_width', 'close_width', 'gripper_force', 'gripper_close_speed', 'grip_timeout', 'move_timeout',
                     'recognize_timeout', 'release_timeout', 'joint_tolerance', 'stable_seconds'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} 必须为有限数值')
            setattr(self, name, value)
        if not 1 <= self.speed_percent <= 20:
            raise ValueError('本试验版速度比例须在 1–20%')
        if not .001 <= self.close_width < self.open_width <= .1:
            raise ValueError('夹爪宽度应为 1 mm ≤ 闭合目标间距 < 张开 ≤ 100 mm；禁止闭合到零')
        if not 0 < self.gripper_force <= 3:
            raise ValueError('夹爪指令参数须在 0–3 N 内；需真机标定验证')
        if not .001 <= self.gripper_close_speed <= .030:
            raise ValueError('夹爪闭合限速须在 1–30 mm/s 内')
        if not all(0 < getattr(self, n) <= 60 for n in
                   ('grip_timeout', 'move_timeout', 'recognize_timeout', 'release_timeout')):
            raise ValueError('超时须在 0–60 秒内')
        if not 0 < self.joint_tolerance <= .1 or not .1 <= self.stable_seconds <= 3:
            raise ValueError('到位容差或稳定时间无效')
        for profile, points in self.points.items():
            if not isinstance(points, dict):
                raise ValueError('点位配置无效')
            for name, value in points.items():
                if name not in ('detect', 'home'):
                    raise ValueError('仅支持检测点和收回点')
                joints(value)


class SettingsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else user_data_root() / 'config.yaml'

    def load(self):
        raw = yaml.safe_load(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        raw = (raw or {}).get('robot', {})
        result = RobotSettings(**{k: v for k, v in raw.items() if k in RobotSettings.__dataclass_fields__})
        result.validate()
        return result

    def save(self, settings):
        settings.validate()
        raw = yaml.safe_load(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        raw = raw or {}
        raw['robot'] = asdict(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='config-', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
