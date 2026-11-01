"""Presentation values populated only from real sensor and inference results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.zodiac import Zodiac


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
    active: bool = False
    area_pct: float = 0.0
    score: float = 0.0
    stability_pct: float = 0.0
    duration_s: float = 0.0
    x: float | None = None
    y: float | None = None
    waveform: tuple[float, ...] = (0.0,) * 30

