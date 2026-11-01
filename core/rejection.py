"""Temporal confirmation cannot turn a rejected model frame into a recognition."""
from __future__ import annotations


class TemporalConfirmation:
    def __init__(self, frames: int = 4) -> None:
        if frames < 1:
            raise ValueError('frames must be positive')
        self.required = frames
        self.reset()

    def reset(self) -> None:
        self.key: str | None = None
        self.count = 0

    def update(self, key: str | None, accepted: bool) -> bool:
        if not accepted or not key:
            self.reset()
            return False
        if key != self.key:
            self.key, self.count = key, 1
        else:
            self.count += 1
        return self.count >= self.required
