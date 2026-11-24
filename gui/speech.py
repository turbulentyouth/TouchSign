"""Decide which fresh recognition results are safe to announce."""

from core.state_machine import RecognitionState
from core.zodiac import ZODIAC_BY_KEY


class SpeechGate:
    def __init__(self, max_frame_gap: float) -> None:
        self.max_frame_gap = max_frame_gap
        self.reset()

    def reset(self) -> None:
        self.last_frame: float | None = None
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.release_since: float | None = None
        self.announced = False
        self.announced_key: str | None = None

    def update(self, state: RecognitionState, timestamp: float) -> str | None:
        if self.last_frame is not None:
            if timestamp == self.last_frame:
                return None
            if not 0 < timestamp - self.last_frame <= self.max_frame_gap:
                self.candidate = None
                self.release_since = None
        self.last_frame = timestamp
        if state.status == 'no_contact':
            self.candidate = None
            if self.release_since is None:
                self.release_since = timestamp
            elif timestamp - self.release_since >= 0.5:
                self.announced = False
            return None
        self.release_since = None
        key = state.prediction.key if state.prediction and state.prediction.accepted else None
        if (self.announced and key == self.announced_key) or key not in ZODIAC_BY_KEY or state.status not in ('preview', 'recognized'):
            self.candidate = None
            return None
        if state.status == 'preview':
            if self.candidate != key:
                self.candidate = key
                self.candidate_since = timestamp
            if timestamp - self.candidate_since < 0.8:
                return None
        elif not state.new_record:
            return None
        self.announced = True
        self.announced_key = key
        return key
