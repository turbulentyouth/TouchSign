"""Dataset capture at jaw arrival, sharing all existing robot motion/stop code."""
from __future__ import annotations

import asyncio
import math
import time

from core.dataset import DATASET_LABELS
from robot.service import RobotService, Stopped


class CollectionRobotService(RobotService):
    def start_collection(self, owner, *, collector, target, interval,
                         stream_id, ready_check, on_result=None):
        # Validate everything before the inherited start can schedule motion.
        self.require_owner(owner)
        if self.busy or self.teaching:
            raise ValueError('请等待本轮结束或退出示教')
        if collector.label not in DATASET_LABELS:
            raise ValueError('请选择与实物一致的标签')
        if collector.label == 'no_contact':
            raise ValueError('空载样本请移开物体后使用“保存一张”，无需运行机械臂')
        if (isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= 20
                or not math.isfinite(interval) or not .1 <= interval <= 60):
            raise ValueError('每轮张数须为 1–20 的整数，间隔须为 0.1–60 秒')
        if collector.calibrating or (not self.mock and not collector.detector.calibrated):
            raise ValueError('请先完成空载校准')
        if collector.recording:
            raise ValueError('当前采集尚未结束')
        self.collector = collector
        self.capture_label = collector.label
        self.capture_target, self.capture_interval = target, interval
        self.capture_count = 0
        self.capture_message = '等待夹爪到位'
        return self.start(owner, ready=self.mock or ready_check(), stream_id=stream_id,
                          ready_check=ready_check, on_result=on_result)

    async def _recognize(self, stream_id, mock_key):
        if self.mock:
            # Exercise the original action sequence without writing fake samples.
            result = await super()._recognize(stream_id, mock_key)
            self.capture_message = '离线模拟完成，未写入采集数据'
            return result
        c = self.collector
        self.recognition_start = time.monotonic()
        c.start(confirmed=True, target=self.capture_target, interval=self.capture_interval,
                max_per_episode=self.capture_target, new_episode=True)
        try:
            while time.monotonic() - self.recognition_start < self.settings.recognize_timeout:
                if self.stop_requested:
                    raise Stopped('已请求停止采集')
                if c.label != self.capture_label:
                    raise RuntimeError('采集中标签发生变化，已停止本轮')
                await self.call('snapshot')
                obs = self._observation(stream_id, after=self.recognition_start)
                self.capture_count = c.saved_run
                self.message = f'采集 {self.capture_count}/{self.capture_target} 张 · {c.message}'
                if obs is None:
                    if time.monotonic() - self.recognition_start > .5:
                        raise RuntimeError('未收到新的触觉采集帧')
                elif c.saved_run >= self.capture_target:
                    return self.capture_label, obs[0]
                if not c.recording:
                    raise RuntimeError(c.message)
                await asyncio.sleep(.05)
            return None, None
        finally:
            # Also runs on stop, disconnect, signal loss and filesystem errors.
            c.stop()
            self.capture_count = c.saved_run
            self.capture_message = f'本轮已保存 {self.capture_count}/{self.capture_target} 张'
            if self.capture_count < self.capture_target:
                self.capture_message += '；未达目标，已保存的数据保留'
