"""TouchSign 界面组件。

所有组件都按 ``gui/preview/index.html`` 的结构与 class 生成原生标签，因此
``gui/style.css`` 可以直接复用。组件只负责展示与局部状态切换，业务数据由
``gui/app.py`` 注入。
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from nicegui import ui

from core.camera import CameraInfo
from core.config import SensorConfig
from core.zodiac import ZODIAC
from gui.view_models import Candidate, ContactReading, Prediction, RecognitionRecord
from gui.dom import Text, element, image, markup, svg

SVG_CAMERA = (
    '<rect x="3.25" y="5.25" width="17.5" height="13.5" rx="2.2"></rect>'
    '<circle cx="12" cy="12" r="3.1"></circle>'
    '<path d="M8 5.25 9.2 3h5.6l1.2 2.25"></path>'
)
SVG_CHEVRON = '<path d="M1 1.25 6 6.25l5-5"></path>'
SVG_REFRESH = (
    '<path d="M20 11a8 8 0 1 0-2.3 5.7"></path>'
    '<path d="M20 4v7h-7"></path>'
)
SVG_LINK = (
    '<path d="M6.4 11.6 11.6 6.4"></path>'
    '<path d="m4.5 9.5-1.2 1.2a3.1 3.1 0 0 0 4.4 4.4l1.8-1.8"></path>'
    '<path d="m13.5 8.5 1.2-1.2a3.1 3.1 0 0 0-4.4-4.4l-1.8 1.8"></path>'
)
SVG_CHECK = '<path d="m2.4 6.1 2.2 2.2 5-5"></path>'


def attr(value: str) -> str:
    """转义要写入 HTML 属性的文本。"""
    return escape(value, quote=True)


def waveform_inner(points: tuple[float, ...]) -> str:
    """把 0..1 的接触力序列绘制成与原型相同的 SVG 波形。"""
    count = len(points)
    step = 240.0 / max(1, count - 1)
    coordinates = [
        (index * step, 29.0 - 6.0 - value * 17.0)
        for index, value in enumerate(points)
    ]
    line = 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in coordinates)
    area = f'{line} L240 30 L0 30 Z'
    return (
        '<defs>'
        '<linearGradient id="wave-stroke" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#4b9f8a" stop-opacity="0.25"></stop>'
        '<stop offset="0.45" stop-color="#78d5b6"></stop>'
        '<stop offset="1" stop-color="#d9bd82"></stop>'
        '</linearGradient>'
        '<linearGradient id="wave-fill" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#78d5b6" stop-opacity="0.14"></stop>'
        '<stop offset="1" stop-color="#78d5b6" stop-opacity="0"></stop>'
        '</linearGradient>'
        '</defs>'
        '<path class="wave-baseline" d="M0 23 H240"></path>'
        f'<path class="wave-area" d="{area}"></path>'
        f'<path class="wave-line" d="{line}"></path>'
    )


class TopBar:
    """顶部品牌、系统状态、时钟与摄像头控制。"""

    def __init__(
        self,
        *,
        on_refresh,
        on_connect,
        on_camera_change,
    ) -> None:
        self._connected = True
        self._busy = False
        self._cameras: tuple[CameraInfo, ...] = ()

        with element('header', 'topbar'):
            with element('div', 'brand'):
                with element('div', 'brand-seal').props('aria-hidden="true"'):
                    Text('span', '触象')
                    element('i')
                with element('div', 'brand-copy'):
                    with element('div', 'brand-title'):
                        Text('h1', '触象')
                        Text('span', 'TOUCHSIGN', 'brand-english')
                    markup(
                        'p',
                        '视触觉生肖识别系统 <span class="micro-separator">/</span> '
                        'TACTILE ZODIAC RECOGNITION',
                    )

            self.system_center = element('div', 'system-center')
            with self.system_center:
                with element('span', 'system-pulse').props('aria-hidden="true"'):
                    element('i')
                with element('div', 'system-copy'):
                    self.system_status = Text('strong', '传感器在线')
                    self.system_subtitle = Text('small', 'SYSTEM READY')
                element('span', 'vertical-rule').props('aria-hidden="true"')
                with element('div', 'system-time'):
                    self.clock = Text('strong', '')
                    self.date = Text('small', '')

            with element('div', 'controls'):
                with element('div', 'sensor-control'):
                    self.control_label = markup(
                        'span', '摄像头 <em>/ CAMERA</em>', 'control-label'
                    )
                    with element('div', 'select-shell') as self.select_shell:
                        svg(SVG_CAMERA, 'sensor-icon')
                        self.select = element('select').props(
                            'aria-label="选择摄像头" disabled '
                            'title="正在搜索摄像头…"'
                        )
                        with self.select:
                            option = Text('option', '正在搜索摄像头…')
                            option.props('value="" disabled selected')
                        svg(SVG_CHEVRON, 'chevron', '0 0 12 8')
                # Native <select> 的事件对象无法通过 args 取到 target.value，
                # 交给页面侧的 js_handler 解构后再发回 Python。
                self.select.on(
                    'change',
                    on_camera_change,
                    js_handler='(event) => emit(event.target.value)',
                )

                self.refresh_button = element('button', 'icon-button').props(
                    'type="button" title="刷新摄像头列表" aria-label="刷新摄像头列表"'
                )
                with self.refresh_button:
                    svg(SVG_REFRESH)
                self.refresh_button.on('click', on_refresh)

                self.connect_button = element('button', 'connect-button').props('type="button"')
                with self.connect_button:
                    element('span', 'connect-dot').props('aria-hidden="true"')
                    self.connect_label = Text('span', '断开连接')
                    svg(SVG_LINK, viewbox='0 0 18 18')
                self.connect_button.on('click', on_connect)

    # ------------------------------------------------------------------ 摄像头
    def set_scanning(self, scanning: bool) -> None:
        self.refresh_button.classes(
            add='is-spinning' if scanning else None,
            remove=None if scanning else 'is-spinning',
        )
        self.select_shell.classes(
            add='is-searching' if scanning else None,
            remove=None if scanning else 'is-searching',
        )

    def set_searching(self) -> None:
        """扫描期间用占位项替换列表，避免用户选中正在释放的设备。"""
        self.select.clear()
        with self.select:
            option = Text('option', '正在搜索摄像头…')
            option.props('value="" disabled selected')
        self.select.props('disabled')
        self.select.props('title="正在搜索摄像头…"')
        self.select_shell.classes(remove='is-empty')
        self.control_label.content = '摄像头 <em>/ CAMERA · 搜索中</em>'

    def set_cameras(
        self,
        cameras: list[CameraInfo],
        selected: int | None = None,
    ) -> None:
        """用扫描结果重建下拉列表。"""
        self._cameras = tuple(cameras)
        self.select.clear()
        self.select_shell.classes(remove='is-searching')

        available = [camera for camera in cameras if camera.available]
        if not available:
            self.select.props('disabled')
            self.select_shell.classes(add='is-empty')
            if cameras:
                text = f'{cameras[0].display_name}（不可用）'
                title = '；'.join(camera.full_label for camera in cameras)
            else:
                text = '未检测到可用摄像头'
                title = '未检测到可用摄像头，请检查连接后点击刷新'
            with self.select:
                option = Text('option', text)
                option.props('value="" disabled selected')
            self.select.props(f'title="{attr(title)}"')
            self.control_label.content = '摄像头 <em>/ CAMERA · 0</em>'
            return

        self.select.props(remove='disabled')
        self.select_shell.classes(remove='is-empty')
        with self.select:
            for camera in cameras:
                option = Text('option', camera.short_label)
                if camera.available:
                    option.props(f'value="{camera.index}"')
                    if camera.index == selected:
                        option.props('selected')
                else:
                    option.props(f'value="__unavailable_{camera.index}" disabled')
        if selected is not None:
            self.select.props(f'value="{selected}"')
        self.select.props(
            f'title="{attr("；".join(camera.full_label for camera in cameras))}"'
        )
        self.control_label.content = (
            f'摄像头 <em>/ CAMERA · {len(available)}</em>'
        )

    def set_selected_camera(self, index: int | None) -> None:
        if index is None:
            return
        self.select.props(f'value="{index}"')

    # ------------------------------------------------------------------ 状态
    def set_clock(self, now: datetime) -> None:
        self.clock.text = now.strftime('%H:%M:%S')
        self.date.text = now.strftime('%Y.%m.%d')

    def set_status(self, text: str, subtitle: str | None = None) -> None:
        self.system_status.text = text
        if subtitle is not None:
            self.system_subtitle.text = subtitle

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        self.connect_button.classes(
            add=None if connected else 'is-disconnected',
            remove='is-disconnected' if connected else None,
        )
        self.system_center.classes(
            add=None if connected else 'is-offline',
            remove='is-offline' if connected else None,
        )
        if not self._busy:
            self.connect_label.text = '断开连接' if connected else '连接传感器'

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.connect_button.classes(
            add='is-connecting' if busy else None,
            remove=None if busy else 'is-connecting',
        )
        if busy:
            self.connect_label.text = '正在连接'
        else:
            self.connect_label.text = '断开连接' if self._connected else '连接传感器'


class SensorPanel:
    """左侧真实传感器图像与接触检测读数。"""

    def __init__(
        self,
        sensor_config: SensorConfig,
        image_url: str,
        reading: ContactReading,
    ) -> None:
        self._sensor_config = sensor_config
        self._demo_url = image_url
        self._camera: CameraInfo | None = None
        self._online = True
        self._streaming = False
        self._feed_rate = 0.0

        with element('section', 'panel sensor-panel').props('aria-labelledby="sensor-heading"'):
            self._build_corners()
            with element('header', 'panel-head'):
                with element('div', 'panel-title'):
                    Text('span', '壹', 'chapter').props('aria-hidden="true"')
                    with element('div', 'title-stack'):
                        markup('span', 'TACTILE VISION <i>/ 光学触觉</i>', 'eyebrow')
                        Text('h2', '视触觉感知').props('id="sensor-heading"')
                with element('div', 'panel-meta'):
                    self.live_chip = element('span', 'live-chip')
                    with self.live_chip:
                        element('i')
                        self.live_text = Text('span', 'NO INPUT')
                    self.camera_meta = markup(
                        'span', 'CAM -- <b>/</b> NO INPUT', 'meta-mono'
                    )

            with element('div', 'sensor-stage'):
                with element('div', 'sensor-view') as self.view:
                    self.image = image(image_url, '等待传感器图像')
                    self._build_overlay_static()
                    self._build_crosshair(reading)
                    self._build_readouts()
                    self._build_feed_note()

            with element('footer', 'sensor-footer'):
                with element('div', 'waveform').props('aria-label="接触变化信号波形"'):
                    markup('span', 'CONTACT SIGNAL <i>· 实时变化</i>', 'wave-caption')
                    self.waveform = svg(
                        waveform_inner(reading.waveform),
                        viewbox='0 0 240 30',
                        preserve_aspect_ratio='none',
                    )
                self.contact_metrics = markup(
                    'div',
                    '<span>尚未校准 <i>/</i> 等待真实读数</span>',
                    'sensor-spec',
                )

        self.set_reading(reading)
        self.show_placeholder('正在搜索摄像头…')

    # ------------------------------------------------------------------ 构建
    @staticmethod
    def _build_corners() -> None:
        for corner in ('tl', 'tr', 'bl', 'br'):
            element('span', f'corner corner--{corner}').props('aria-hidden="true"')

    def _build_overlay_static(self) -> None:
        ui.html(
            '<span class="feed-vignette"></span>'
            '<span class="scan-line"></span>'
            '<span class="feed-grain"></span>'
            '<div class="hud-corners"><i></i><i></i><i></i><i></i></div>'
        )
        with element('div', 'feed-head'):
            with element('div', 'feed-title'):
                Text('span', '01', 'tiny-index')
                markup('span', '接触面实时图像<small>CONTACT SURFACE</small>')
            self.feed_code = markup(
                'span',
                'NO INPUT',
                'feed-code',
            )
        ui.html(
            '<div class="target-label"><span></span> TACTILE SENSOR</div>'
        )

    def _build_crosshair(self, reading: ContactReading) -> None:
        with element('div', 'crosshair').props('aria-hidden="true"'):
            element('span', 'crosshair-h')
            element('span', 'crosshair-v')
            element('span', 'crosshair-core')
            self.coordinate = markup('span', self._coordinate_text(reading), 'coordinate')

    def _build_readouts(self) -> None:
        with element('div', 'feed-readout'):
            self.area = self._readout('变化面积', 'AREA', '0.0', '%')
            self.score = self._readout('接触变化', 'DELTA', '0.0', '')
            self.duration = self._readout('接触时长', 'TIME', '0.0', 's')

    def _build_feed_note(self) -> None:
        self.feed_note = element('div', 'feed-note')
        with self.feed_note:
            element('i')
            self.feed_note_text = Text('span', '')

    @staticmethod
    def _readout(label: str, english: str, initial: str, unit: str) -> ui.html:
        with element('div', 'readout-item'):
            markup('span', f'{label} <small>{english}</small>')
            return markup('strong', f'{initial}<em>{unit}</em>')

    @staticmethod
    def _coordinate_text(reading: ContactReading) -> str:
        if reading.x is None or reading.y is None:
            return 'X ----&nbsp;&nbsp;Y ----'
        return f'X {reading.x:07.1f}&nbsp;&nbsp;Y {reading.y:07.1f}'

    # ------------------------------------------------------------------ 图像源
    def show_stream(
        self,
        url: str,
        camera: CameraInfo,
        *,
        pending: bool = False,
    ) -> None:
        """切换到摄像头实时图像（MJPEG 流）。

        ``pending`` 表示采集线程尚未输出第一帧，此时用 LINK 状态提示
        “正在打开摄像头”，由页面定时器在拿到帧后切换为 LIVE。
        """
        self._streaming = True
        self._feed_rate = 0.0
        self.view.classes(add='is-live')
        self.image.props(f'src="{attr(url)}" alt="摄像头实时采集的触觉图像"')
        self.set_camera(camera)
        self._set_chip('LINK' if pending else 'LIVE', 'demo' if pending else 'live')
        self._set_note('正在打开摄像头…' if pending else '')
        self._update_feed_code()

    def show_placeholder(self, note: str = '') -> None:
        """没有实时帧时显示空白占位，绝不回退到生肖示例图。"""
        self._streaming = False
        self._feed_rate = 0.0
        self.view.classes(remove='is-live')
        self.image.props(f'src="{attr(self._demo_url)}" alt="等待传感器图像"')
        self._set_chip('OFFLINE' if not self._online else 'NO INPUT',
                       'offline' if not self._online else 'demo')
        self._set_note(note)
        self._update_feed_code()

    def set_camera(self, camera: CameraInfo | None) -> None:
        self._camera = camera
        if camera is None:
            self.camera_meta.content = 'CAM -- <b>/</b> NO INPUT'
            self.camera_meta.props('title="未连接摄像头"')
        else:
            self.camera_meta.content = (
                f'CAM {camera.index:02d} <b>/</b> {camera.resolution}'
            )
            self.camera_meta.props(f'title="{attr(camera.full_label)}"')
        self._update_feed_code()

    def set_signal(self, healthy: bool) -> None:
        if not self._streaming or not self._online:
            return
        if healthy:
            self._set_chip('LIVE', 'live')
            self._set_note('')
        else:
            self._set_chip('NO SIGNAL', 'offline')

    def set_feed_rate(self, fps: float) -> None:
        self._feed_rate = max(0.0, float(fps))
        self._update_feed_code()

    # ------------------------------------------------------------------ 状态
    def _set_chip(self, text: str, kind: str) -> None:
        self.live_text.text = text
        self.live_chip.classes(remove='is-offline is-demo')
        if kind == 'demo':
            self.live_chip.classes(add='is-demo')
        elif kind == 'offline':
            self.live_chip.classes(add='is-offline')

    def _set_note(self, text: str) -> None:
        self.feed_note_text.text = text
        self.feed_note.classes(
            add='is-visible' if text else None,
            remove=None if text else 'is-visible',
        )

    def _feed_code_html(self) -> str:
        if self._streaming and self._camera is not None:
            camera = self._camera
            fps = self._feed_rate if self._feed_rate > 0 else camera.fps
            rate = (
                f'<b>{fps:.1f}</b> <i>FPS</i>'
                if fps > 0
                else '<b>--</b> <i>FPS</i>'
            )
            return ' <i>·</i> '.join([camera.resolution, rate])
        return 'OFFLINE' if not self._online else 'NO INPUT'

    def _update_feed_code(self) -> None:
        self.feed_code.content = self._feed_code_html()

    def set_reading(self, reading: ContactReading) -> None:
        self.area.content = f'{reading.area_pct:.1f}<em>%</em>'
        self.score.content = f'{reading.score:.2f}'
        self.duration.content = f'{reading.duration_s:.1f}<em>s</em>'
        self.coordinate.content = self._coordinate_text(reading)
        self.waveform.content = waveform_inner(reading.waveform)
        self.contact_metrics.content = (
            f'<span>变化面积 <b>{reading.area_pct:.1f}%</b></span>'
            '<span class="spec-divider"></span>'
            f'<span>接触 <b>{reading.duration_s:.1f}s</b></span>'
        )

    def set_connected(self, connected: bool) -> None:
        self._online = connected
        if connected:
            self._set_chip('LIVE' if self._streaming else 'NO INPUT',
                           'live' if self._streaming else 'demo')
            self._update_feed_code()
            return

        self.show_placeholder('传感器已断开 · 摄像头采集已停止')
        for item in (self.area, self.score, self.duration):
            item.content = '—'
        self.coordinate.content = 'X ----&nbsp;&nbsp;Y ----'
        self.waveform.content = waveform_inner(tuple(0.04 for _ in range(30)))
        self._set_note('传感器已断开 · 摄像头采集已停止')


class CandidateRow:
    def __init__(self, rank: int) -> None:
        self.row = element('div', 'candidate-row')
        with self.row:
            Text('span', f'{rank:02d}', 'candidate-rank')
            self.name = markup('span', '', 'candidate-name')
            with element('span', 'candidate-track'):
                self.bar = element('i')
            self.value = markup('span', '', 'candidate-value')

    def update(self, candidate: Candidate, *, leading: bool) -> None:
        self.name.content = (
            f'{candidate.zodiac.glyph} <small>{candidate.zodiac.title}</small>'
        )
        self.bar.style(f'width: {candidate.confidence:.1f}%')
        self.value.content = f'{candidate.confidence:.1f}<em>%</em>'
        self.row.classes(
            add='is-leading' if leading else None,
            remove=None if leading else 'is-leading',
        )


class ResultPanel:
    """右侧辨识结果面板。"""

    def __init__(
        self,
        threshold: float,
        prediction: Prediction | None,
        records: list[RecognitionRecord],
        stability: float,
    ) -> None:
        self._threshold = threshold

        with element('aside', 'panel result-panel').props('aria-labelledby="result-heading"'):
            self._build_corners()
            with element('header', 'panel-head'):
                with element('div', 'panel-title'):
                    Text('span', '贰', 'chapter').props('aria-hidden="true"')
                    with element('div', 'title-stack'):
                        markup('span', 'IDENTIFICATION <i>/ 生肖辨识</i>', 'eyebrow')
                        Text('h2', '辨识结果').props('id="result-heading"')
                with element('div', 'panel-meta'):
                    self.result_state = element('span', 'result-state')
                    with self.result_state:
                        element('i')
                        self.result_state_text = Text('span', '结果稳定')

            with element('div', 'result-body'):
                with element('div', 'result-main'):
                    self._build_oracle()
                    self._build_copy(prediction)

                markup(
                    'div',
                    '<span></span><i>◆</i><span></span>',
                    'engraved-divider',
                ).props('aria-hidden="true"')

                with element('div', 'result-stats'):
                    self.sequence = self._stat('生肖序列', 'SEQUENCE')
                    self.confidence = self._stat('模型置信度', 'CONFIDENCE')
                    self.latency = self._stat('推理耗时', 'LATENCY')

                with element('section', 'recent').props('aria-labelledby="recent-heading"'):
                    with element('div', 'recent-head'):
                        markup('h3', '最近识别 <span>RECENT LOG</span>').props(
                            'id="recent-heading"'
                        )
                        self.recent_count = markup('span', '', 'recent-count')
                    with element('div', 'log-list') as self.log_list:
                        pass

            with element('footer', 'result-footer'):
                markup('span', '录', 'record-seal').props('aria-hidden="true"')
                self.record_status = Text('span', '结果已写入识别记录')
                self.record_id = Text('span', '', 'record-id')

        if prediction is None:
            self.render_unknown()
        else:
            self.render(prediction, stability, animate=False)
        self.render_records(records)

    @staticmethod
    def _build_corners() -> None:
        for corner in ('tl', 'tr', 'bl', 'br'):
            element('span', f'corner corner--{corner}').props('aria-hidden="true"')

    def _build_oracle(self) -> None:
        with element('div', 'oracle').props('aria-hidden="true"'):
            element('span', 'oracle-aura')
            element('span', 'oracle-line oracle-line--v')
            element('span', 'oracle-line oracle-line--h')
            element('span', 'orbit orbit--outer')
            element('span', 'orbit orbit--dashed')
            element('span', 'orbit-ticks')
            self.oracle_top = markup('span', '', 'oracle-caption oracle-caption--top')
            self.oracle_bottom = markup('span', '', 'oracle-caption oracle-caption--bottom')
            self.seal = element('span', 'result-seal')
            with self.seal:
                with element('span', 'seal-inner'):
                    self.glyph = Text('strong', '')
                    Text('small', '生肖印')
                markup('span', '识', 'seal-stamp')

    def _build_copy(self, prediction: Prediction | None) -> None:
        with element('div', 'result-copy'):
            self.eyebrow = markup('span', '', 'eyebrow result-eyebrow')
            with element('div', 'name-line'):
                self.title = Text('h2', '')
                self.match_chip = markup('span', '<i></i> MATCHED', 'match-chip')
            self.latin = markup('p', '', 'result-latin')
            self.poem = Text('p', '', 'result-poem')

    @staticmethod
    def _stat(label: str, english: str) -> ui.html:
        with element('div', 'stat'):
            markup('span', f'{label} <small>{english}</small>')
            return markup('strong', '')

    def render(
        self,
        prediction: Prediction,
        stability: float,
        *,
        animate: bool = True,
        preview: bool = False,
    ) -> None:
        if not prediction.accepted:
            self.render_unknown('置信度不足，暂无有效识别结果')
            return
        zodiac = prediction.zodiac
        value = prediction.confidence

        self.glyph.text = zodiac.glyph
        self.title.text = zodiac.title
        self.eyebrow.content = (
            f'ZODIAC NO. <b>{zodiac.display_index}</b> '
            f'<i>/</i> <span>{zodiac.branch}</span>'
        )
        self.latin.content = (
            f'<span>{zodiac.pinyin}</span> <i>·</i> <span>{zodiac.english}</span> '
            f'<em>|</em> <span>{zodiac.nature}</span>'
        )
        self.poem.text = zodiac.poem
        self.oracle_top.content = f'{zodiac.direction_en} <i>·</i> {zodiac.element}'
        self.oracle_bottom.content = f'{zodiac.hour} <i>·</i> {zodiac.display_index}'


        self.sequence.content = f'<b>{zodiac.display_index}</b><em>/ 12</em>'
        self.confidence.content = f'<b>{value:.1f}</b><em>%</em>'
        self.latency.content = f'<b>{prediction.latency_ms:.1f}</b><em>ms</em>'
        self.record_id.text = f'RECORD #{prediction.record_id}'

        if preview:
            self._mark_preview()
        else:
            self.record_status.text = '结果已写入识别记录'
            self.result_state.classes(remove='is-rejected')
            self.result_state_text.text = '结果稳定'
            self.match_chip.classes(remove='is-rejected')
            self.match_chip.content = '<i></i> MATCHED'

        if animate:
            self._animate()

    def _mark_preview(self) -> None:
        self.result_state.classes(remove='is-rejected')
        self.result_state_text.text = '模型预览'
        self.match_chip.classes(remove='is-rejected')
        self.match_chip.content = '<i></i> TOP 1'
        self.record_status.text = '当前帧的模型首选预测'
        self.record_id.text = ''

    def render_negative_preview(self, label: str, reason: str, confidence: float, latency_ms: float) -> None:
        self.render_unknown(reason)
        self.title.text = label
        self.eyebrow.content = 'MODEL <i>/</i> TOP 1'
        self.latin.content = ''
        self.confidence.content = f'<b>{100 * confidence:.1f}</b><em>%</em>'
        self.latency.content = f'<b>{latency_ms:.1f}</b><em>ms</em>'
        self._mark_preview()

    def render_unknown(self, reason: str = '等待有效目标，请放置待识别物体') -> None:
        self.glyph.text = '—'
        self.title.text = '未识别'
        self.eyebrow.content = '等待辨识 / NO RESULT'
        self.latin.content = '暂无有效生肖结果'
        self.poem.text = reason
        self.oracle_top.content = ''
        self.oracle_bottom.content = ''
        self.sequence.content = '—'
        self.confidence.content = '—'
        self.latency.content = '—'
        self.result_state_text.text = '未识别'
        self.result_state.classes(add='is-rejected')
        self.match_chip.classes(add='is-rejected')
        self.match_chip.content = '<i></i> NO RESULT'
        self.record_status.text = '暂无有效结果，未写入识别记录'
        self.record_id.text = ''

    def set_stability(self, value: float) -> None:
        pass  # Stability is no longer displayed in the result panel.

    def _animate(self) -> None:
        self.seal.classes(add='is-updating')
        self.title.classes(add='is-updating')
        ui.timer(0.55, self._clear_animation, once=True)

    def _clear_animation(self) -> None:
        self.seal.classes(remove='is-updating')
        self.title.classes(remove='is-updating')

    def render_records(self, records: list[RecognitionRecord]) -> None:
        self.recent_count.content = f'<i></i> {len(records):02d}'
        self.log_list.clear()
        with self.log_list:
            for record in records:
                self._log_row(record)

    @staticmethod
    def _log_row(record: RecognitionRecord) -> None:
        signet_class = 'log-signet'
        if record.zodiac.key == 'dragon':
            signet_class += ' log-signet--dragon'
        elif record.zodiac.key == 'horse':
            signet_class += ' log-signet--horse'

        with element('div', 'log-row'):
            Text('span', record.timestamp.strftime('%H:%M:%S'), 'log-time')
            Text('span', record.zodiac.glyph, signet_class).props('aria-hidden="true"')
            markup(
                'span',
                f'{record.zodiac.title} <small>{record.zodiac.english}</small>',
                'log-name',
            )
            markup('span', f'{record.confidence:.1f}<em>%</em>', 'log-confidence')
            svg(SVG_CHECK, 'log-check', '0 0 12 12')


class ZodiacMatrix:
    """底部十二生肖矩阵。"""

    def __init__(self, selected: str) -> None:
        self.cards: dict[str, ui.element] = {}
        with element('section', 'zodiac-section').props('aria-labelledby="zodiac-heading"'):
            with element('div', 'zodiac-heading'):
                Text('span', '十二', 'zodiac-seal').props('aria-hidden="true"')
                with element('div', 'title-stack'):
                    markup('span', 'ZODIAC MATRIX <i>/ 天干地支</i>', 'eyebrow')
                    Text('h2', '十二生肖').props('id="zodiac-heading"')
                element('span', 'heading-rule').props('aria-hidden="true"')
                with element('div', 'zodiac-legend'):
                    with element('span', 'legend-active'):
                        element('i')
                        Text('span', '感应高亮')
                    Text('span', '根据识别结果自动高亮', 'legend-muted')

            with element('div', 'zodiac-row'):

                for zodiac in ZODIAC:
                    card = element('div', 'zodiac-card').props(
                        f'data-zodiac="{zodiac.key}" '
                        f'aria-label="生肖{zodiac.glyph}的识别状态"'
                    )
                    self.cards[zodiac.key] = card
                    with card:
                        with element('span', 'zodiac-top'):
                            Text('span', zodiac.display_index, 'zodiac-index')
                            Text('span', zodiac.branch, 'zodiac-branch')
                        Text('span', zodiac.glyph, 'zodiac-char')
                        with element('span', 'zodiac-foot'):
                            Text('span', zodiac.title, 'zodiac-cn')
                            Text('span', zodiac.english, 'zodiac-en')
                        element('span', 'match-mark').props('aria-hidden="true"')

        self.set_active(selected)

    def set_active(self, key: str) -> None:
        for card_key, card in self.cards.items():
            active = card_key == key
            card.classes(
                add='is-active' if active else None,
                remove=None if active else 'is-active',
            )
            card.props(f'aria-current="{str(active).lower()}"')


class AppFooter:
    def __init__(self) -> None:
        with element('footer', 'app-footer'):
            markup('span', 'TOUCHSIGN <i>/</i> 触象', 'footer-brand')
            element('span', 'footer-rule').props('aria-hidden="true"')
            Text('span', '观象于触，识形于微', 'footer-motto')
            element('span', 'footer-rule footer-rule--right').props('aria-hidden="true"')
            Text('span', 'TACTILE INTELLIGENCE · ONNX CPU', 'footer-system')
