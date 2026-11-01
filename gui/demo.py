"""界面演示数据源。

在真实相机、接触检测与 ONNX 推理接入之前，由本模块生成稳定的演示数据，
让界面可以完整呈现“接触 → 识别 → 记录”的状态变化。所有随机量都使用固定
种子，便于截图对比和回归检查。

接入真实业务时，把 ``TouchSignView`` 中的数据源替换为 ``core.camera`` /
``core.inference`` 的输出即可，组件本身不需要改动。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.config import Config
from core.zodiac import ZODIAC, ZODIAC_BY_KEY, Zodiac


@dataclass(frozen=True, slots=True)
class Candidate:
    zodiac: Zodiac
    confidence: float


@dataclass(frozen=True, slots=True)
class Prediction:
    zodiac: Zodiac
    confidence: float
    candidates: tuple[Candidate, ...]
    latency_ms: float
    accepted: bool
    record_id: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class RecognitionRecord:
    zodiac: Zodiac
    confidence: float
    timestamp: datetime
    record_id: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class ContactReading:
    active: bool
    area_pct: float
    force_n: float
    temp_c: float
    stability_pct: float
    duration_s: float
    x: float
    y: float
    waveform: tuple[float, ...]


class DemoPresenter:
    """生成演示用的接触读数与识别结果。"""

    def __init__(self, config: Config, *, seed: int = 20261030) -> None:
        self.config = config
        self._rng = random.Random(seed)
        initial = config.recognition.initial_zodiac
        self._cursor = ZODIAC_BY_KEY.get(initial, ZODIAC[0]).index

    def contact(self, *, active: bool = True) -> ContactReading:
        """返回一帧带轻微抖动的接触读数。"""
        if not active:
            return ContactReading(
                active=False, area_pct=0.0, force_n=0.0, temp_c=0.0,
                stability_pct=0.0, duration_s=0.0, x=0.0, y=0.0,
                waveform=tuple(0.05 for _ in range(30)),
            )

        phase = self._rng.random() * math.tau
        waveform: list[float] = []
        value = 0.5
        for index in range(30):
            value += self._rng.uniform(-0.13, 0.13)
            value = min(0.92, max(0.1, value))
            ripple = 0.035 * math.sin(index * 0.62 + phase)
            waveform.append(round(min(0.98, max(0.03, value + ripple)), 4))

        return ContactReading(
            active=True,
            area_pct=round(42.6 + self._rng.uniform(-0.35, 0.35), 1),
            force_n=round(1.24 + self._rng.uniform(-0.03, 0.03), 2),
            temp_c=round(26.4 + self._rng.uniform(-0.05, 0.05), 1),
            stability_pct=round(94.2 + self._rng.uniform(-0.6, 0.6), 1),
            duration_s=round(1.4 + self._rng.uniform(-0.15, 0.35), 1),
            x=round(328.6 + self._rng.uniform(-0.8, 0.8), 1),
            y=round(412.9 + self._rng.uniform(-0.8, 0.8), 1),
            waveform=tuple(waveform),
        )

    def initial_prediction(self) -> Prediction:
        zodiac = ZODIAC_BY_KEY.get(self.config.recognition.initial_zodiac, ZODIAC[0])
        return self.prediction_for(
            zodiac,
            confidence=zodiac.baseline_confidence,
            record_id='A7F2',
            timestamp=datetime.now(),
        )

    def prediction_for(
        self,
        zodiac: Zodiac,
        *,
        confidence: float | None = None,
        record_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> Prediction:
        value = round(zodiac.baseline_confidence if confidence is None else confidence, 1)
        candidates = self._candidates(zodiac, value)
        return Prediction(
            zodiac=zodiac,
            confidence=value,
            candidates=candidates,
            latency_ms=round(18.0 + self._rng.uniform(-1.5, 2.5), 1),
            accepted=value / 100.0 >= self.config.recognition.confidence_threshold,
            record_id=record_id or self._new_record_id(),
            timestamp=timestamp or datetime.now(),
        )

    def next_prediction(self) -> Prediction:
        """按十二生肖顺序推进一次自动识别。"""
        self._cursor = self._cursor % len(ZODIAC) + 1
        zodiac = ZODIAC[self._cursor - 1]
        confidence = zodiac.baseline_confidence + self._rng.uniform(-0.3, 0.3)
        return self.prediction_for(zodiac, confidence=round(min(99.4, max(95.0, confidence)), 1))

    def initial_history(self) -> list[RecognitionRecord]:
        """最近识别面板的初始四条记录（与视觉原型一致）。"""
        now = datetime.now()
        seeded = (
            ('dragon', 97.4, 251),
            ('horse', 96.8, 408),
            ('rabbit', 96.1, 607),
            ('monkey', 95.8, 863),
        )
        return [
            RecognitionRecord(
                zodiac=ZODIAC_BY_KEY[key],
                confidence=confidence,
                timestamp=now - timedelta(seconds=seconds),
                record_id=self._new_record_id(),
                accepted=confidence / 100.0 >= self.config.recognition.confidence_threshold,
            )
            for key, confidence, seconds in seeded
        ]

    def _candidates(self, zodiac: Zodiac, confidence: float) -> tuple[Candidate, ...]:
        residual = max(0.1, 100.0 - confidence)
        following = ZODIAC[zodiac.index % len(ZODIAC)]
        third = ZODIAC[(zodiac.index + 1) % len(ZODIAC)]
        values = (
            confidence,
            max(0.2, residual * 0.48),
            max(0.1, residual * 0.2),
        )
        return tuple(
            Candidate(item, round(value, 1))
            for item, value in zip((zodiac, following, third), values)
        )

    def _new_record_id(self) -> str:
        return f'{self._rng.randrange(0x1000, 0x10000):04X}'
