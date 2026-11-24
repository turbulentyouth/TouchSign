"""Robot controls share the existing recognition screen."""
import asyncio
import json
from nicegui import ui
from core.camera import camera_manager
from core.zodiac import ZODIAC_BY_KEY
from robot.settings import SettingsStore, RobotSettings, joints
from robot.service import RobotService

_service = None


def get_service():
    global _service
    if _service is None:
        store = SettingsStore()
        try:
            settings = store.load()
        except Exception as exc:
            # Do not overwrite a malformed user file.
            settings = RobotSettings()
            ui.notify(f'应用配置读取失败：{exc}', type='negative')
        # A fresh application always starts in offline mode, even if hardware
        # was selected last time. Each connection is deliberate.
        settings.backend = 'mock'
        _service = RobotService(settings, store)
    return _service


async def shutdown_robot():
    if _service and _service.owner is not None:
        await _service.disconnect(_service.owner)


class RobotPanel:
    def __init__(self, view):
        self.view = view
        self.service = get_service()
        self.result_latched = False
        self.action_busy = False
        self.detect_button = view.detect_button
        self.stop_button = view.robot_stop_button
        s = self.service.settings
        with ui.card().classes('robot-panel w-full').style('background:#111923;color:#e5dfd3;border:1px solid #a68a4b; padding:12px 18px'):
            with ui.row().classes('items-center w-full gap-3'):
                ui.label('NERO · 单次触觉检测').classes('text-lg font-bold')
                self.mode = ui.select({'mock':'离线模拟（不控制硬件）','nero':'NERO 真机（未验证）'}, value='mock', label='设备模式').classes('w-64')
                self.channel = ui.input('USB-CAN 通道', value=s.channel).classes('w-28')
                self.connect_button = ui.button('连接', on_click=lambda: self.dispatch(self.connect))
                self.disconnect_button = ui.button('断开', on_click=lambda: self.dispatch(self.disconnect))
            self.real_ack = ui.checkbox('真机：已核对固件、夹爪及负载，确认直达路径无障碍；首次使用须低速验证', value=False)
            self.notice = ui.label('离线模拟：没有真实机械臂或传感器；模拟结果不计入识别记录').classes('text-amber-400')
            with ui.row().classes('items-center gap-2'):
                self.teach_button = ui.button('开始重力补偿示教', on_click=lambda: self.dispatch(self.enter_teach))
                self.detect_point = ui.button('保存检测点', on_click=lambda: self.dispatch(lambda: self.save_point('detect')))
                self.home_point = ui.button('保存收回点', on_click=lambda: self.dispatch(lambda: self.save_point('home')))
                self.end_teach = ui.button('结束示教 / 保持', on_click=lambda: self.dispatch(self.exit_teach))
            with ui.row().classes('items-center gap-2') as self.mock_controls:
                self.drag_angles = ui.input('模拟拖动：七关节角 rad', value='[0, 0.2, -0.2, 0, 0.1, 0, 0]').classes('w-96')
                self.drag_button = ui.button('应用模拟姿态', on_click=lambda: self.dispatch(self.drag))
                self.mock_key = ui.select({**{k:v.glyph for k,v in ZODIAC_BY_KEY.items()}, 'timeout':'模拟识别超时'}, value='rat', label='模拟触觉结果').classes('w-44')
            with ui.expansion('夹爪与速度设置（点位与参数同存应用配置）').classes('w-full'):
                with ui.row():
                    self.speed = ui.number('速度比例 %', value=s.speed_percent, min=1, max=20, step=1)
                    self.open_width = ui.number('张开宽度 mm', value=s.open_width*1000, min=1,max=100)
                    self.close_width = ui.number('压牌闭合目标 mm', value=s.close_width*1000,min=0,max=99)
                    self.force = ui.number('夹爪指令 N（待真机核验）', value=s.gripper_force,min=.1,max=3,step=.1)
            self.points_label = ui.label('检测点：未保存 · 收回点：未保存')
            self.status = ui.label('未连接')
        ui.timer(.15, self.refresh)
        self.refresh()

    @property
    def locked(self):
        return self.action_busy or self.service.busy or self.service.teaching

    async def dispatch(self, action):
        if self.action_busy and action not in (self.stop, self.disconnect):
            return
        self.action_busy = True
        try:
            await action()
        except Exception as exc:
            ui.notify(str(exc), type='negative', timeout=8000)
        finally:
            self.action_busy = False
            self.refresh()

    def apply_settings(self):
        if self.service.owner is not None and self.service.owner is not self:
            raise RuntimeError('设备由另一个窗口控制')
        from dataclasses import replace
        old = self.service.settings
        new = replace(old, backend=self.mode.value, channel=str(self.channel.value),
                      speed_percent=int(self.speed.value), open_width=float(self.open_width.value)/1000,
                      close_width=float(self.close_width.value)/1000, gripper_force=float(self.force.value))
        new.validate()
        self.service.store.save(new)
        self.service.settings = new
        if self.service.arm:
            self.service.arm.settings = new

    async def connect(self):
        if self.mode.value == 'nero' and not self.real_ack.value:
            raise ValueError('请先核对真机安装条件并勾选确认；当前包尚未真机验证')
        self.apply_settings()
        await self.service.connect(self)

    async def disconnect(self):
        await self.service.disconnect(self)

    async def enter_teach(self):
        self.apply_settings()
        await self.service.teach(self, True)

    async def exit_teach(self):
        await self.service.teach(self, False)

    async def save_point(self, name):
        await self.service.save_point(self, name)

    async def drag(self):
        self.service.require_owner(self)
        if not self.service.mock or self.service.busy:
            raise ValueError('仅离线示教可应用模拟姿态')
        await self.service.call('simulate_drag', joints(json.loads(self.drag_angles.value)))

    async def stop(self):
        await self.service.stop(self)

    async def detect(self):
        v = self.view
        if v.calibrating or v.loading_model or v.scanning or v.speech_testing:
            raise ValueError('请先结束校准、加载或语音试听')
        self.apply_settings()
        packet = camera_manager.frame_packet()
        import time
        ready = self.service.mock or (v.streaming and v.connected and v.engine.model is not None
                 and v.engine.detector.calibrated and packet is not None
                 and time.monotonic()-packet.timestamp < v.config.recognition.max_frame_age)
        # Validate before clearing the previous result; start sets busy immediately.
        client = ui.context.client
        def report(key, state, simulated):
            with client:
                self.result_latched = True
                if key is None:
                    v.result.render_unknown('本轮未获得稳定生肖，正在松开并收回')
                    v.matrix.set_active('')
                    return
                if simulated:
                    v.result.render_speech_test(ZODIAC_BY_KEY[key])
                    v.result.result_state_text.text = '离线模拟 · 非真实识别'
                    v.result.record_status.text = '模拟流程结果，不写入识别记录'
                    v.result.match_chip.content = '<i></i> SIMULATED'
                    v.matrix.set_active(key)
                else:
                    v._render_state(state, force=True)
                if v.speech_enabled:
                    source = json.dumps(f'/assets/speech/{key}.wav')
                    ui.run_javascript(f'queueRecognitionSpeech(getHtmlElement({v.audio.id}), {source})')
        self.service.start(self, ready=bool(ready), stream_id=v._stream_serial,
                           mock_key=self.mock_key.value, on_result=report,
                           ready_check=lambda: self.sensor_ready())
        self.result_latched = False
        v.speech.reset()
        v._clear_result('本轮检测运行中', '检测中')

    def sensor_ready(self):
        import time
        v = self.view
        packet = camera_manager.frame_packet()
        return (v.connected and v.streaming and not v.calibrating and v.engine.detector.calibrated
                and packet is not None and packet.serial == v._stream_serial
                and time.monotonic()-packet.timestamp < v.config.recognition.max_frame_age)

    def refresh(self):
        s = self.service
        own = s.owner is self
        idle = own and not s.busy and not self.action_busy
        self.status.set_text(f'#{s.cycle_id} · {s.phase} · {s.message}')
        points = s.points if own else {}
        self.points_label.set_text(' · '.join(f'{label}：' + ('已保存' if key in points else '未保存') for key,label in [('detect','检测点'),('home','收回点')]))
        self.connect_button.set_enabled(s.owner is None and not self.action_busy)
        self.disconnect_button.set_enabled(own and not self.action_busy)
        self.stop_button.set_enabled(own)
        self.teach_button.set_enabled(idle and s.phase in ('idle','complete'))
        self.end_teach.set_enabled(idle and s.teaching)
        self.detect_point.set_enabled(idle and s.phase == 'teaching')
        self.home_point.set_enabled(idle and s.phase == 'teaching')
        self.drag_button.set_enabled(idle and s.mock and s.phase == 'teaching')
        self.detect_button.set_enabled(idle and s.phase in ('idle','complete') and len(points)==2)
        self.mode.set_enabled(s.owner is None)
        self.channel.set_enabled(s.owner is None)
        self.real_ack.set_visibility(self.mode.value == 'nero')
        self.mock_controls.set_visibility(self.mode.value == 'mock')
        self.notice.set_text('离线模拟：不控制硬件，结果不计入识别记录' if self.mode.value=='mock' else '真机接口未实测：先验证重力补偿和直达路径，再运行压牌检测')
        for control in (self.speed,self.open_width,self.close_width,self.force):
            control.set_enabled(not s.busy and not s.teaching and not self.action_busy)
        for name in ('calibration_button','reload_button','speech_test_button'):
            element = getattr(self.view,name,None)
            if element:
                if s.busy or s.teaching:
                    element.props('disabled')
                else:
                    element.props(remove='disabled')
