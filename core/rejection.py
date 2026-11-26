"""Temporal confirmation cannot turn a rejected model frame into a recognition."""
from __future__ import annotations

# Shared by continuous manual testing and the robot's per-cycle preview gate.
PREVIEW_CONFIRM_FRAMES = 3
PREVIEW_CONFIRM_SECONDS = .15


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


class PreviewConfirmation(TemporalConfirmation):
    """Require matching fresh frames over the same interval as robot previews."""

    def __init__(self, max_frame_gap: float) -> None:
        self.max_frame_gap = max_frame_gap
        super().__init__(PREVIEW_CONFIRM_FRAMES)

    def reset(self) -> None:
        super().reset()
        self.since = 0.0
        self.last_timestamp: float | None = None

    def update(self, key: str | None, accepted: bool, timestamp: float) -> bool:
        if self.last_timestamp is not None:
            if timestamp == self.last_timestamp:
                return (accepted and key == self.key and self.count >= self.required
                        and timestamp - self.since >= PREVIEW_CONFIRM_SECONDS - 1e-9)
            if not 0 < timestamp - self.last_timestamp <= self.max_frame_gap:
                self.reset()
        if not accepted or not key:
            self.reset()
            return False
        if key != self.key:
            self.since = timestamp
        self.last_timestamp = timestamp
        enough_frames = super().update(key, accepted)
        return enough_frames and timestamp - self.since >= PREVIEW_CONFIRM_SECONDS - 1e-9
