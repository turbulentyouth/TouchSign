"""Manual raw-image capture, without calibration, inference or robot control."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from core.camera import FramePacket
from core.dataset import DATASET_LABELS, _encode_png

FORMAT = 'touchsign_manual_raw_v1'
MAX_FRAME_AGE = .5


class ManualDatasetWriter:
    """Single-worker writer. Paths in samples.jsonl are relative to the session."""

    def __init__(self, root: str | Path, *, camera: dict | None = None) -> None:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.session_dir = Path(root).expanduser().resolve() / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._history: list[dict] = []
        metadata = {
            'format': FORMAT, 'session_id': session_id,
            'created_at': datetime.now().astimezone().isoformat(),
            'camera': camera or {}, 'labels': list(DATASET_LABELS),
            'rotation_degrees': 180, 'input_kind': 'raw_bgr_v1',
            'label_source': 'manual', 'calibration': None,
        }
        (self.session_dir / 'session.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
        )

    def save(self, packet: FramePacket, label: str, captured_at: str, *,
             correction: dict | None = None) -> dict:
        if label not in DATASET_LABELS:
            raise ValueError(f'不支持的标签：{label!r}')
        # CameraStream._publish already rotates camera pixels by 180 degrees.
        # Encode the packet directly: rotating here again would undo it.
        encoded = _encode_png(packet.frame)
        folder = self.session_dir / label
        folder.mkdir(exist_ok=True)
        image_file = folder / (uuid.uuid4().hex + '.png')
        row = {
            'format': FORMAT, 'image': image_file.relative_to(self.session_dir).as_posix(),
            'label': label, 'label_source': 'manual', 'captured_at': captured_at,
            'rotation_degrees': 180, 'frame_shape': list(packet.frame.shape),
            'stream_serial': packet.serial, 'frame_sequence': packet.sequence,
        }
        if correction is not None:
            row['correction'] = correction
        line = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
        # Roll back failed appends so a disk error cannot leave a successful
        # metadata row pointing at a missing or partially written PNG.
        with (self.session_dir / 'samples.jsonl').open('a+b') as manifest:
            position = manifest.tell()
            try:
                with image_file.open('xb') as image:
                    image.write(encoded)
                    image.flush()
                    os.fsync(image.fileno())
                manifest.write(line)
                manifest.flush()
                os.fsync(manifest.fileno())
            except Exception:
                try:
                    manifest.truncate(position)
                    manifest.flush()
                finally:
                    image_file.unlink(missing_ok=True)
                raise
        self._history.append(row)
        return row

    def undo(self) -> dict | None:
        """Remove the latest saved row, then archive its image outside the dataset.

        Called on the same worker as save(), so queued captures and undo requests
        have one well-defined order. The manifest replacement is the commit point:
        it must never reference an image which has already been moved away.
        """
        if not self._history:
            return None
        row = self._history[-1]
        manifest = self.session_dir / 'samples.jsonl'
        lines = manifest.read_bytes().splitlines(keepends=True)
        if not lines or json.loads(lines[-1]) != row:
            raise RuntimeError('采集清单已被其他程序修改，未撤回任何图片')
        image_file = self.session_dir / row['image']
        archive = self.session_dir / '_undone' / row['image']
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Retain the original label and timestamp even after it leaves the manifest.
        archive.with_suffix('.json').write_text(json.dumps({
            'sample': row, 'undone_at': datetime.now().astimezone().isoformat(),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        fd, temporary = tempfile.mkstemp(prefix='.samples-', suffix='.tmp', dir=self.session_dir)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.writelines(lines[:-1])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self._history.pop()
        try:
            image_file.replace(archive)
        except OSError as exc:
            # The sample is already excluded from training. Do not report an undo
            # failure or leave the counter/history inconsistent after that commit.
            return {**row, 'undo_warning': f'已从训练清单移除；原图归档失败，仍在原位置：{exc}'}
        return row


class ManualCollector:
    """Keep disk I/O off the preview thread; bound queued full-resolution frames."""

    def __init__(self, writer: ManualDatasetWriter, *, label: str | None = None,
                 interval: float = .2) -> None:
        if label is not None and label not in DATASET_LABELS:
            raise ValueError('请选择有效标签')
        if not .1 <= interval <= 60:
            raise ValueError('连拍间隔须为 0.1–60 秒')
        self.writer = writer
        self.label = label
        self.interval = interval
        self.counts: Counter[str] = Counter()
        self.recording = False
        self.error: str | None = None
        self.message = '选标签后手工按压，空格保存；Z / 退格撤回，R 连拍，Esc 停止。'
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='manual-save')
        self._pending: deque[tuple[str, Future]] = deque()
        self._last_key: tuple[int, int] | None = None
        self._serial: int | None = None
        self._next_shot = 0.

    @property
    def pending(self) -> int:
        return len(self._pending)

    @staticmethod
    def fresh(packet: FramePacket | None) -> bool:
        return packet is not None and 0 <= time.monotonic() - packet.timestamp <= MAX_FRAME_AGE

    def choose_label(self, label: str) -> None:
        if label not in DATASET_LABELS:
            raise ValueError('请选择有效标签')
        if label != self.label:
            self.stop()
            self.label = label
            self.message = '标签已切换，按空格保存或 R 开始连拍。'

    def stop(self) -> None:
        self.recording = False
        self.message = '已停止连拍，按空格保存或 R 继续。'

    def _ready(self, packet: FramePacket | None) -> bool:
        if self.error:
            return False
        if self.label is None:
            self.message = '先按 1–0、-、=、U 或 N 选择标签。'
            return False
        if not self.fresh(packet):
            self.recording = False
            self.message = '没有新鲜图像，请检查传感器连接。'
            return False
        return True

    def start(self, packet: FramePacket | None) -> None:
        if self.recording or not self._ready(packet):
            return
        self.recording = True
        self._serial = packet.serial
        self._next_shot = 0.
        self.message = '连拍中，请保持按压；松开或换物体前按 Esc。'

    def capture_once(self, packet: FramePacket | None) -> bool:
        self._collect_finished()
        if not self._ready(packet):
            return False
        key = (packet.serial, packet.sequence)
        if key == self._last_key:
            if not self.recording:
                self.message = '当前帧已保存，请等待下一帧。'
            return False
        if self.pending >= 4:
            self.message = '正在写入图像，请稍候。'
            return False
        # Freeze both pixels and label at the trigger, not when the worker runs.
        snapshot = replace(packet, frame=packet.frame.copy())
        captured_at = datetime.fromtimestamp(
            time.time() - (time.monotonic() - packet.timestamp),
        ).astimezone().isoformat(timespec='milliseconds')
        self._pending.append(('save', self._worker.submit(self.writer.save, snapshot, self.label, captured_at)))
        self._last_key = key
        return True

    def undo(self) -> bool:
        self.stop()
        self._collect_finished()
        if self.error:
            return False
        if self.pending >= 8:
            self.message = '待处理操作较多，请稍候再撤回。'
            return False
        self._pending.append(('undo', self._worker.submit(self.writer.undo)))
        self.message = '正在撤回最近一张；已停止连拍。'
        return True

    def _collect_finished(self) -> None:
        while self._pending and self._pending[0][1].done():
            operation, future = self._pending.popleft()
            if future.cancelled():
                continue
            try:
                row = future.result()
                if operation == 'undo':
                    if row is None:
                        self.message = '本会话没有可撤回的图片。'
                        continue
                    self.counts[row['label']] -= 1
                    if not self.counts[row['label']]:
                        del self.counts[row['label']]
                    if self._last_key == (row['stream_serial'], row['frame_sequence']):
                        self._last_key = None
                    self.message = (row.get('undo_warning') or
                                    f"已撤回 {row['label']}，剩余 {sum(self.counts.values())} 张；选正确标签后补拍。")
                else:
                    self.counts[row['label']] += 1
                    self.message = f"已保存 {row['label']}，本会话共 {sum(self.counts.values())} 张。"
            except Exception as exc:
                self.recording = False
                name = '撤回' if operation == 'undo' else '保存'
                self.error = f'{name}失败：{exc}。请排除原因后重新启动。'
                for _, pending in self._pending:
                    pending.cancel()

    def poll(self, packet: FramePacket | None):
        self._collect_finished()
        if not self.fresh(packet):
            if self.recording:
                self.stop()
                self.message = '传感器信号中断，已停止连拍；恢复后按 R 继续。'
            return None
        if self._serial is not None and packet.serial != self._serial:
            self.stop()
            self.message = '传感器已重新连接，按空格保存或 R 继续。'
        self._serial = packet.serial
        now = time.monotonic()
        if self.recording and now >= self._next_shot:
            if self.capture_once(packet):
                self._next_shot = now + self.interval
        return packet.frame

    def close(self) -> None:
        self.recording = False
        self._worker.shutdown(wait=True)
        self._collect_finished()
