"""Continuous recognition with an explicit human-labelled correction dialog."""
from __future__ import annotations

import base64

import cv2
from nicegui import run, ui

from core.config import resolve_path
from core.corrections import CorrectionRecorder
from core.zodiac import ZODIAC_BY_KEY, ZODIAC_KEYS
from gui.app import TouchSignView


class ManualTestView(TouchSignView):
    def __init__(self, config, *, output=None, **kwargs):
        self.corrections = CorrectionRecorder(
            output or resolve_path('data/corrections', writable=True), config.recognition.max_frame_age)
        self.correction_snapshot = None
        self.saving_correction = False
        super().__init__(config, **kwargs)
        self._build_correction_dialog()
        with self.runtime_controls:
            self.correction_button = ui.button('保存纠错样本', on_click=self._open_correction,
                                               icon='add_photo_alternate').props('no-caps')
            self.correction_count = ui.label('纠错样本：0 张').classes('text-xs')

    def _build_correction_dialog(self):
        with ui.dialog().props('persistent') as self.correction_dialog:
            with ui.card().classes('w-[760px] max-w-full').style('max-height: 90vh; overflow-y: auto'):
                ui.label('标注当前按压图像').classes('text-lg font-bold')
                ui.label('图像与预测已冻结，可以松开实物。请核对图像后选择正确生肖。').classes('text-sm')
                self.correction_image = ui.image().classes('w-full').style('max-height: 340px').props('fit=contain')
                self.correction_prediction = ui.label().classes('text-sm')
                self.correction_label = ui.select(
                    {key: ZODIAC_BY_KEY[key].glyph for key in ZODIAC_KEYS},
                    label='实物的正确生肖（必选）', value=None,
                ).classes('w-full')
                self.correction_notes = ui.input('按压位置 / 姿态备注（可选）',
                                                 placeholder='例如：右上角、向左倾斜时误判').classes('w-full')
                self.correction_error = ui.label().classes('text-red-400 text-sm')
                with ui.row():
                    self.correction_save = ui.button('按正确标签保存', on_click=self._save_correction)
                    self.correction_cancel = ui.button('取消', on_click=self._cancel_correction).props('flat')

    def _on_processed_frame(self, packet, state):
        self.corrections.observe(packet, state, generation=self._generation,
                                 model=self.engine.model, camera=self.camera,
                                 threshold=self.config.recognition.confidence_threshold)

    def _open_correction(self):
        if self.correction_snapshot is not None or self.saving_correction:
            return
        if (not self.connected or not self.streaming or self._pending_reset or self.signal_lost
                or self.calibrating or self.scanning or self.loading_model or self.speech_testing):
            ui.notify('请先完成传感器连接和空载校准，保持按压并等待识别', type='warning')
            return
        try:
            snapshot = self.corrections.freeze(generation=self._generation, stream_serial=self._stream_serial)
            # JPEG is only for the dialog; the dataset writer saves the original PNG.
            frame = snapshot.packet.frame
            ratio = min(1., 700 / max(frame.shape[:2]))
            preview = cv2.resize(frame, (round(frame.shape[1] * ratio), round(frame.shape[0] * ratio)))
            ok, encoded = cv2.imencode('.jpg', preview)
            if not ok:
                raise ValueError('无法显示冻结图像，请重试')
            self.correction_image.set_source('data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii'))
        except Exception as exc:
            ui.notify(str(exc), type='warning')
            return
        prediction = snapshot.details['prediction']
        key = prediction['predicted_key']
        label = ZODIAC_BY_KEY[key].glyph if key in ZODIAC_BY_KEY else {'unknown': '未知物体', 'no_contact': '空载'}.get(key, key)
        self.correction_prediction.set_text(
            f'该帧模型预测：{label} · {prediction["confidence"]:.1%} · 帧 {snapshot.packet.sequence}')
        self.correction_label.value = None
        self.correction_notes.value = ''
        self.correction_error.set_text('')
        self.correction_snapshot = snapshot
        self.correction_dialog.open()

    def _cancel_correction(self):
        if not self.saving_correction:
            self.correction_snapshot = None
            self.correction_dialog.close()

    async def _save_correction(self):
        if self.saving_correction or self.correction_snapshot is None:
            return
        if self.correction_label.value not in ZODIAC_KEYS:
            self.correction_error.set_text('请选择实物的正确生肖')
            return
        self.saving_correction = True
        self.correction_error.set_text('')
        controls = (self.correction_save, self.correction_cancel, self.correction_label, self.correction_notes)
        for control in controls:
            control.disable()
        try:
            row = await run.io_bound(self.corrections.save, self.correction_snapshot,
                                     self.correction_label.value, self.correction_notes.value or '')
            if row is None:
                return
            self.correction_count.set_text(f'纠错样本：{self.corrections.count} 张')
            self.correction_count.tooltip(row['saved_path'])
            ui.notify(f'已保存正确标签「{ZODIAC_BY_KEY[row["label"]].glyph}」的纠错样本', type='positive')
            self.correction_snapshot = None
            self.correction_dialog.close()
        except Exception as exc:
            self.correction_error.set_text(f'保存失败：{exc}。冻结图像已保留，可重试。')
        finally:
            self.saving_correction = False
            for control in controls:
                control.enable()
