"""Contact-only adapter for the existing GUI and robot observation protocol."""
from __future__ import annotations

import math
import time

from core.camera import FramePacket
from core.collection import CollectionController
from core.state_machine import RecognitionState


class CollectionEngine:
    """Use the collector's reference and contact detector; never load a model."""

    model = None
    model_status = '数据采集 · 人工标注'

    def __init__(self, controller: CollectionController, max_frame_age: float = .5):
        self.controller = controller
        self.model_path = controller.writer.root
        self.max_frame_age = min(.5, max_frame_age)

    @property
    def detector(self):
        return self.controller.detector

    def reset(self, clear_calibration=True):
        if clear_calibration:
            self.controller.invalidate('设备已变化，请重新空载校准并选择标签。')

    def process(self, frame, *, sequence, timestamp, stream_id=0):
        c = self.controller
        with c.lock:
            if (frame is None or timestamp is None or not math.isfinite(timestamp)
                    or not -.1 <= time.monotonic() - timestamp <= self.max_frame_age):
                message = '无实时传感器信号，请重新连接并空载校准。'
                if c.detector.calibrated or c.recording:
                    c.invalidate(message)
                else:
                    c.message = message
                return RecognitionState('no_signal', c.message)
            c.poll_packet(FramePacket(stream_id, sequence, timestamp, frame))
            contact = c.contact
            if not c.detector.calibrated or contact is None:
                return RecognitionState('uncalibrated', c.message)
            status = ('unknown' if not contact.valid else 'no_contact' if not contact.active
                      else 'contact' if contact.stable else 'unstable')
            return RecognitionState(status, contact.reason, contact)
