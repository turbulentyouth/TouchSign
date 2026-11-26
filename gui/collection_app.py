"""A collection variant of the Ubuntu console, with the original robot controls."""
from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time

from nicegui import app, run, ui

from core.camera import camera_manager
from core.collection import CollectionController
from core.collection_engine import CollectionEngine
from core.dataset import DATASET_LABELS
from core.zodiac import ZODIAC_BY_KEY
from gui.app import TouchSignView, LOGGER
from gui.dom import Text, element
from gui.robot_panel import RobotPanel
from gui.view_models import ContactReading
from robot.collection_service import CollectionRobotService

LABELS = {**{key: item.glyph for key, item in ZODIAC_BY_KEY.items()},
          'unknown': '未知物体', 'no_contact': '空载 / 无接触'}


class CollectionPanel:
    def __init__(self, view):
        self.view = view
        with element('aside', 'panel collection-panel'):
            with element('header', 'panel-head'):
                with element('div', 'title-stack'):
                    Text('span', 'DATA COLLECTION / 人工标注', 'eyebrow')
                    Text('h2', '数据采集')
            with ui.column().classes('collection-body w-full gap-3'):
                ui.label('① 空载校准  →  ② 选择标签  →  ③ 运行一轮').classes('text-sm text-amber-200')
                self.label = ui.select(LABELS, label='实物标签（必选）', value=None,
                                       on_change=view._choose_label).classes('w-full')
                with ui.row().classes('w-full gap-3'):
                    self.target = ui.number('每轮目标张数', value=view.default_target, min=1,
                                            max=20, step=1).classes('flex-1')
                    self.interval = ui.number('最短间隔 / 秒', value=view.default_interval,
                                              min=.1, max=60, step=.1).classes('flex-1')
                self.capture = ui.button('保存一张（手动补采）', icon='add_a_photo',
                                         on_click=view._capture_once).classes('w-full').props('no-caps')
                ui.label('机械臂到位并接触稳定后自动采集；默认每轮 1 张。空载类请移开物体后手动保存。').classes('text-xs text-gray-400')
                self.status = ui.label('请移开物体，点击“确认空载并校准”。').classes('collection-status')
                self.counts = ui.label('当前会话：0 张').classes('text-lg text-amber-200')
                self.per_label = ui.label('尚未保存样本').classes('text-sm')
                with ui.expansion('会话与保存位置', value=True).classes('w-full'):
                    self.notes = ui.input('会话备注（下次校准时生效）', value=view.initial_notes,
                                          placeholder='操作者、牌子批次、姿态、光照等').classes('w-full')
                    self.session = ui.label('尚未建立会话').classes('text-xs break-all')
                    ui.label(str(view.collector.writer.root)).classes('text-xs break-all')
                    ui.label('保存原图、空载基准、差分图和标签清单；每次校准新建独立会话。').classes('text-xs text-gray-400')

    def refresh(self):
        v, c = self.view, self.view.collector
        locked = v.collection_locked
        for control in (self.label, self.target, self.interval, self.notes):
            control.set_enabled(not locked)
        self.capture.set_enabled(not locked and c.detector.calibrated and c.label in DATASET_LABELS)
        if c.label is None and self.label.value is not None:
            self.label.value = None
        counts = c.writer.counts.copy()
        self.counts.set_text(f'当前会话：{sum(counts.values())} 张 · 当前标签：{counts[c.label]} 张')
        self.per_label.set_text(' · '.join(f'{LABELS[key]} {counts[key]}' for key in DATASET_LABELS if counts[key])
                                or '尚未保存样本')
        self.session.set_text('会话：' + (c.writer.session_id or '尚未建立；先空载校准'))


class CollectionRobotPanel(RobotPanel):
    def __init__(self, view):
        super().__init__(view, service_factory=CollectionRobotService)
        self.mock_key.set_visibility(False)

    async def detect(self):
        v = self.view
        v.validate_label()
        if v.calibrating or v.scanning or v.manual_busy:
            raise ValueError('请先等待校准、扫描或保存完成')
        target = float(v.result.target.value)
        if not target.is_integer():
            raise ValueError('每轮张数须为整数')
        self.apply_settings()
        client = ui.context.client

        def report(key, state, simulated):
            with client:
                v.result.status.set_text(self.service.capture_message)
                v.result.refresh()

        self.service.start_collection(
            self, collector=v.collector,
            target=int(target), interval=float(v.result.interval.value),
            stream_id=v._stream_serial, ready_check=self.sensor_ready, on_result=report,
        )
        v.result.status.set_text('本轮运行中：等待夹爪到位后采集。')

    def refresh(self):
        super().refresh()
        v, s = self.view, self.service
        self.detect_button.set_text('采集（运行一轮）')
        if hasattr(v, 'result'):
            self.detect_button.set_enabled(
                self.detect_button.enabled and not v.calibrating and not v.scanning and not v.manual_busy
                and v.collector.label in DATASET_LABELS and v.collector.label != 'no_contact'
                and (s.mock or self.sensor_ready()))
            if s.owner is self and s.phase == 'complete' and hasattr(s, 'capture_message'):
                capture_status = (s.capture_message if s.result is not None
                                  else f'本轮已停止，已保存 {s.capture_count} 张')
                v.runtime_hint.text = s.message + ' · ' + capture_status
        self.notice.set_text('离线模拟：动作流程相同，不写入采集数据' if self.mode.value == 'mock'
                             else '人工标签采集：夹爪到位、接触稳定后保存，再松开并收回')


class CollectionView(TouchSignView):
    def __init__(self, config, *, output: Path, notes='', target=1, interval=.5):
        self.collector = CollectionController(camera_manager, output, contact_config=config.contact)
        self.initial_notes, self.default_target, self.default_interval = notes, target, interval
        self.manual_busy = False
        self.cancel_calibration = threading.Event()
        super().__init__(config, engine=CollectionEngine(self.collector, config.recognition.max_frame_age),
                         robot_panel_factory=CollectionRobotPanel,
                         result_panel_factory=lambda *args: CollectionPanel(self))
        self.app.classes(add='collection-app')
        self.topbar.subtitle.content = '视触觉数据采集与标注 / TACTILE DATA COLLECTION'
        self.matrix.legend.text = '当前标签'
        self.matrix.legend_hint.text = '根据人工选择的实物标签高亮'
        self.footer.system.text = 'TACTILE DATA · LOSSLESS PNG'
        for key, card in self.matrix.cards.items():
            card.props(f'aria-label="采集标签：{LABELS[key]}"')
        ui.add_css('''
            .collection-panel { min-width: 0; overflow: auto; }
            .collection-panel .panel-head { min-height: 62px; }
            .collection-body { padding: 18px 24px; }
            .collection-status { width: 100%; padding: 10px 12px; border-left: 2px solid #d4ad5e;
                background: #d4ad5e12; color: #e9d9af; min-height: 44px; }
            .collection-app .workspace { grid-template-columns: minmax(0, 1.15fr) minmax(380px, 1fr); }
            @media (max-width: 900px) { .collection-app .workspace { grid-template-columns: 1fr; } }
        ''')
        self.difference_preview = False
        self.preview_text.text = self._preview_label
        self.result.status.set_text('请移开物体，点击“确认空载并校准”。')
        self.result.refresh()
        ui.timer(.2, self.result.refresh)
        app.on_shutdown(self._stop_collection)

    @property
    def collection_locked(self):
        return (self.calibrating or self.scanning or self.manual_busy
                or (self.robot_panel is not None and self.robot_panel.locked))

    def _build_runtime_controls(self):
        with element('div', 'runtime-controls'):
            self.calibration_button = element('button', 'runtime-button').props('type="button"')
            with self.calibration_button:
                self.calibration_text = Text('span', '确认空载并校准')
            self.calibration_button.on('click', self._calibrate)
            if self.config.app.robot_controls:
                self.detect_button = ui.button('采集（运行一轮）', on_click=self._detect_robot).classes('runtime-detect')
                self.detect_button.disable()
                self.robot_stop_button = ui.button('停止本轮并收回', on_click=self._stop_robot).classes('runtime-stop')
                self.robot_stop_button.disable()
                self.robot_settings_toggle = ui.button('展开机械臂设置', on_click=self._toggle_robot_settings).classes('runtime-settings-toggle').props('flat no-caps aria-expanded=false')
            self.exit_button = ui.button('退出程序', on_click=app.shutdown, icon='power_settings_new').classes('runtime-exit')
            self.preview_button = element('button', 'runtime-button').props('type="button"')
            with self.preview_button:
                self.preview_text = Text('span', self._preview_label)
            self.preview_button.on('click', self._toggle_preview)
            Text('span', '数据采集 · 人工标注', 'runtime-model-status')
            self.runtime_hint = Text('span', '校准前请移开所有物体', 'runtime-hint')

    def _update_model_status(self):
        pass  # This console has no model or inference controls.

    def _clear_result(self, message, status=''):
        self.current = None
        self.result.status.set_text(message)
        self.matrix.set_active(self.collector.label or '')

    def _choose_label(self, event):
        if self.collection_locked:
            self.result.label.value = self.collector.label
            return
        self.collector.choose_label(event.value)
        self.result.status.set_text('请选择与实物一致的标签，选择后即可采集。')
        self.matrix.set_active(self.collector.label or '')

    def validate_label(self):
        if self.collector.label not in DATASET_LABELS:
            raise ValueError('请选择与实物一致的标签')

    async def _capture_once(self):
        if self.collection_locked:
            return
        self.manual_busy = True
        try:
            self.validate_label()
            while self.analyzing:
                await asyncio.sleep(.01)
            saved = await run.io_bound(self.collector.capture_once, expected_label=self.collector.label)
            self.result.status.set_text(self.collector.message)
            if not saved:
                ui.notify(self.collector.message, type='warning')
        except Exception as exc:
            ui.notify(str(exc), type='negative')
        finally:
            self.manual_busy = False
            self.result.refresh()

    async def _calibrate(self):
        if self.collection_locked:
            return
        if not self.streaming or not self.connected:
            ui.notify('请先连接传感器', type='warning')
            return
        self.calibrating = True
        self.cancel_calibration.clear()
        self._generation += 1
        generation = self._generation
        self.calibration_button.props('disabled')
        self.calibration_text.text = '校准中…'
        self.result.status.set_text('正在采集空载基准，请保持表面无物体。')
        try:
            while self.analyzing:
                await asyncio.sleep(.01)
            await run.io_bound(self.collector.calibrate, confirmed_unloaded=True,
                               notes=self.result.notes.value or '', cancel=self.cancel_calibration)
            packet = camera_manager.frame_packet()
            if (generation != self._generation or packet is None
                    or packet.serial != self.collector._selected_serial
                    or not -.1 <= time.monotonic() - packet.timestamp <= self.engine.max_frame_age):
                raise ValueError('校准结束时设备已变化，请重新校准')
            self._stream_serial = packet.serial
            self._pending_reset = False
            self._last_token = (0, 0)
            self.result.label.value = None
            self.result.status.set_text('空载校准完成，已新建会话。请选择实物标签后开始采集。')
            self.topbar.set_status('空载校准完成', 'READY TO COLLECT')
        except Exception as exc:
            self.collector.invalidate(f'校准失败：{exc}')
            self.result.status.set_text(self.collector.message)
            ui.notify(self.collector.message, type='negative')
        finally:
            self.calibrating = False
            self.calibration_button.props(remove='disabled')
            self.calibration_text.text = '确认空载并校准'
            self.result.refresh()

    async def _tick(self):
        if self.analyzing or self.calibrating or self.scanning or self.manual_busy:
            return
        self.analyzing = True
        generation = self._generation
        try:
            if self._pending_reset:
                await run.io_bound(self.engine.reset)
                self._pending_reset = False
            packet = camera_manager.frame_packet() if self.streaming and self.connected else None
            if packet is not None and packet.serial != self._stream_serial:
                self._stream_serial = packet.serial
                await run.io_bound(self.engine.reset)
                self._last_token = (0, 0)
            fresh = packet is not None and -.1 <= time.monotonic() - packet.timestamp <= self.engine.max_frame_age
            panel = self.robot_panel
            service = panel.service if panel else None
            if fresh:
                self._render_preview(packet.frame)
                # No contact processing or writes during approach/grip/return or simulation.
                if service and service.busy and (service.owner is not panel or service.mock
                                                or service.phase not in ('recognizing', 'releasing')):
                    return
                if service and service.phase == 'recognizing' and packet.timestamp <= service.recognition_start:
                    return
                token = packet.serial, packet.sequence
                if token == self._last_token:
                    return
                self._last_token = token
            state = await run.io_bound(self.engine.process, packet.frame if fresh else None,
                                       sequence=packet.sequence if fresh else 0,
                                       timestamp=packet.timestamp if fresh else None,
                                       stream_id=self._stream_serial)
            if generation != self._generation:
                return
            if service and service.owner is panel and not service.mock and packet is not None:
                service.observe(state, packet.timestamp, packet.serial)
            contact = state.contact
            self._waveform.append(contact.score)
            self.sensor.set_reading(ContactReading(
                active=contact.active, area_pct=100 * contact.area_fraction,
                score=contact.score, stability_pct=100 * contact.stability,
                duration_s=contact.duration_s, x=contact.center_x, y=contact.center_y,
                waveform=tuple(self._waveform)))
            self.topbar.set_status(state.message, 'DATA COLLECTION')
            if not fresh:
                self.sensor.show_placeholder('无实时信号，请连接传感器并重新空载校准')
            if self.collector.recording:
                self.result.status.set_text(self.collector.message)
            elif not self.collector.detector.calibrated:
                self.result.status.set_text(self.collector.message)
        except Exception as exc:
            LOGGER.exception('采集帧处理失败')
            self.collector.invalidate(f'采集失败：{exc}')
            self.result.status.set_text(self.collector.message)
        finally:
            self.analyzing = False

    def _stop_collection(self):
        self.cancel_calibration.set()
        self.collector.invalidate('采集已停止，请重新连接并空载校准。')

    def _stop_stream(self):
        self._stop_collection()
        super()._stop_stream()

    def _on_disconnect(self):
        self._stop_collection()
        self.connected = self.streaming = False
        super()._on_disconnect()
