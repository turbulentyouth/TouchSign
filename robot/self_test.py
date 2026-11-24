"""Packaged offline acceptance check. Never opens a CAN device or camera."""
import asyncio
import json
from pathlib import Path
import selectors
import tempfile
import time
import wave
from robot.settings import RobotSettings, SettingsStore
from robot.service import RobotService
from robot.transport import can_interface


class _Selector(selectors.SelectSelector):
    def select(self, timeout=None):
        time.sleep(min(.01, timeout) if timeout is not None else .01)
        return []


class _Loop(asyncio.SelectorEventLoop):
    # Self-test has no socket I/O. Avoid Windows loopback socket restrictions.
    def __init__(self):
        super().__init__(_Selector())
    def _make_self_pipe(self): pass
    def _close_self_pipe(self): pass
    def _write_to_self(self): pass


async def _cycle(root):
    s=RobotService(RobotSettings(stable_seconds=.1),SettingsStore(root/'config.yaml'))
    owner=object()
    try:
        await s.connect(owner)
        await s.teach(owner,True)
        await s.call('simulate_drag',[.1]*7)
        await s.save_point(owner,'detect')
        await s.call('simulate_drag',[0.]*7)
        await s.save_point(owner,'home')
        await s.teach(owner,False)
        reports=[]
        await s.start(owner,ready=True,on_result=lambda *r:reports.append(r))
        assert s.phase=='complete' and s.completed==1 and len(reports)==1
        assert s.arm.q==(0.,)*7 and s.arm.width==.06
        assert s.store.load().points['mock']['detect']==[.1]*7
        return {'phase':s.phase,'completed':s.completed,'result':reports[0][0],'simulated':True}
    finally:
        if s.arm:
            await s.disconnect(owner)
        s.pool.shutdown(wait=True)


def run_self_test(report_path):
    from core.config import PROJECT_ROOT, load_config
    from core.inference import OnnxClassifier
    from core.zodiac import ZODIAC_KEYS
    from can.interface import _get_class_for_interface
    from pyAgxArm import create_agx_arm_config, AgxArmFactory
    import numpy as np
    root=PROJECT_ROOT
    report={'hardware_opened':False}
    try:
        config=load_config(root/'config/config.yaml')
        model=OnnxClassifier(root/'models/raw/best.onnx',preview=True)
        pred=model.predict(np.zeros((64,64,3),dtype=np.uint8))
        report['onnx_inference']=pred.predicted_key
        for key in ZODIAC_KEYS:
            with wave.open(str(root/'assets/speech'/f'{key}.wav'),'rb') as f:
                assert f.getnframes()>0
        report['speech_files']=len(ZODIAC_KEYS)
        interface = can_interface()
        _get_class_for_interface(interface)
        report['can_interface'] = interface
        for version in ('default','v111','v112','v120','v121'):
            AgxArmFactory.create_arm(create_agx_arm_config(
                robot='nero', firmeware_version=version, interface=interface,
                channel=RobotSettings().channel, auto_connect=False, enable_check_can=False))
        report['sdk_firmware_profiles']=5
        if interface == 'agx_cando':
            from agx_cando.dll import load_cando_dll
            from agx_cando.native import _load_receive_api
            try:
                load_cando_dll()
                _load_receive_api()
                report['native_can_dll']='loaded; ABI 2'
            except Exception as exc:
                report['native_can_dll']=str(exc)
                report['hardware_limitation']=True
        else:
            report['native_can_dll']='not required (Linux SocketCAN)'
        with tempfile.TemporaryDirectory() as tmp, asyncio.Runner(loop_factory=_Loop) as runner:
            report['cycle']=runner.run(_cycle(Path(tmp)))
        report['offline_checks_passed']=True
    except Exception as exc:
        report['offline_checks_passed']=False
        report['error']=repr(exc)
    Path(report_path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['offline_checks_passed'] else 1
