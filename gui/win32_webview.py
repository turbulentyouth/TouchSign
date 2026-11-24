"""Win32 + signed Microsoft WebView2Loader, without .NET or custom native EXEs.

COM method indices/GUIDs match Microsoft.Web.WebView2.Core.Raw metadata from
the pinned SDK shipped by pywebview. Only IUnknown-based baseline interfaces
are used. All COM calls occur on the STA/window thread.
"""
import base64
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
import uuid

HRESULT=C.c_long
PTR=C.c_void_p
CALL=C.WINFUNCTYPE
LRESULT=C.c_ssize_t
IUnknown=uuid.UUID('00000000-0000-0000-c000-000000000046').bytes_le


def com(pointer, index, result, types, *args):
    table=C.cast(pointer,C.POINTER(C.POINTER(PTR))).contents
    return CALL(result,PTR,*types)(table[index])(pointer,*args)


def check(result, operation):
    if result < 0:
        raise OSError(f'{operation}: HRESULT 0x{result & 0xffffffff:08x}')
    return result


class Handler:
    def __init__(self, iid, types, callback, fail):
        self.iid=uuid.UUID(iid).bytes_le
        self.refs=1
        def query(this, guid, out):
            if C.string_at(guid,16) in (IUnknown,self.iid):
                out[0]=C.cast(C.pointer(self.obj),PTR).value
                self.refs+=1
                return 0
            out[0]=None
            return -2147467262
        def addref(this):
            self.refs+=1
            return self.refs
        def release(this):
            self.refs=max(0,self.refs-1)
            return self.refs
        def invoke(this,*args):
            try:
                callback(*args)
                return 0
            except Exception:
                fail(traceback.format_exc())
                return -2147467259
        self.callbacks=[CALL(HRESULT,PTR,PTR,C.POINTER(PTR))(query),
            CALL(W.ULONG,PTR)(addref),CALL(W.ULONG,PTR)(release),CALL(HRESULT,PTR,*types)(invoke)]
        self.table=(PTR*4)(*[C.cast(f,PTR).value for f in self.callbacks])
        class Object(C.Structure):
            _fields_=[('vtable',C.POINTER(PTR))]
        self.obj=Object(C.cast(self.table,C.POINTER(PTR)))
        self.pointer=C.cast(C.pointer(self.obj),PTR)


WNDPROC=CALL(LRESULT,W.HWND,W.UINT,W.WPARAM,W.LPARAM)
class WNDCLASSEX(C.Structure):
    _fields_=[('cbSize',W.UINT),('style',W.UINT),('lpfnWndProc',WNDPROC),
        ('cbClsExtra',C.c_int),('cbWndExtra',C.c_int),('hInstance',W.HINSTANCE),
        ('hIcon',W.HICON),('hCursor',W.HANDLE),('hbrBackground',W.HBRUSH),
        ('lpszMenuName',W.LPCWSTR),('lpszClassName',W.LPCWSTR),('hIconSm',W.HICON)]


class Win32WebView:
    def __init__(self, html_path=None, profile=None, on_message=None, on_ready=None,
                 selftest_script=None, report=None, loader_path=None, on_created=None):
        self.html_path=Path(html_path).resolve() if html_path else None
        self.profile=Path(profile or Path(os.environ.get('LOCALAPPDATA','.'))/'TouchSign/webview2').resolve()
        self.on_message=on_message or (lambda value:None)
        self.on_ready=on_ready
        self.on_created=on_created
        self.script=Path(selftest_script).read_text(encoding='utf-8') if selftest_script else None
        self.report_path=Path(report) if report else None
        self.report={}
        self.loader_path=Path(loader_path) if loader_path else self._find_loader()
        self.commands=queue.Queue()
        self.handlers=[]
        self.environment=self.controller=self.web=None
        self.hwnd=None
        self.ready=False
        self.finished=False
        self.deadline=time.monotonic()+60
        self.thread_id=None

    @staticmethod
    def _find_loader():
        base=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent.parent))
        candidates=[base/'native-window/WebView2Loader.dll',
            base/'.venv-runtime/Lib/site-packages/webview/lib/runtimes/win-x64/native/WebView2Loader.dll']
        for path in candidates:
            if path.is_file(): return path
        raise FileNotFoundError('Microsoft WebView2Loader.dll not found')

    def _handler(self,iid,types,callback):
        handler=Handler(iid,types,callback,self._fail)
        self.handlers.append(handler)
        return handler.pointer

    def _emit(self,value):
        self.on_message(value)

    def _fail(self,message):
        if self.finished: return
        self.report={'passed':False,'type':'host-error','message':str(message)}
        self._emit(self.report)
        if self.report_path:
            self.report_path.write_text(json.dumps(self.report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.finished=True
        if not self.script:
            self.user.MessageBoxW(self.hwnd, str(message), 'TouchSign 窗口启动失败',0x10)
        self.close()

    def post(self,value):
        self.commands.put(('post',value))

    def close(self):
        self.commands.put(('close',None))

    def execute(self,script,callback=None):
        if threading.get_ident()!=self.thread_id:
            self.commands.put(('execute',(script,callback)))
            return
        def complete(result,text):
            check(result,'ExecuteScript')
            if callback: callback(text)
        handler=self._handler('49511172-cc67-4bca-9923-137112f4c4cc',[HRESULT,W.LPCWSTR],complete)
        check(com(self.web,29,HRESULT,[W.LPCWSTR,PTR],script,handler),'ExecuteScript request')

    def _resize(self):
        if self.controller and self.hwnd:
            bounds=W.RECT()
            self.user.GetClientRect(self.hwnd,C.byref(bounds))
            check(com(self.controller,6,HRESULT,[W.RECT],bounds),'Controller.Bounds')

    def _environment_ready(self,result,environment):
        self._emit({'type':'host-progress','stage':'environment-callback','hresult':result})
        check(result,'CreateCoreWebView2Environment')
        self.environment=environment
        com(environment,1,W.ULONG,[])
        handler=self._handler('6c4819f3-c9b7-4260-8127-c9f5bde7f68c',[HRESULT,PTR],self._controller_ready)
        check(com(environment,3,HRESULT,[W.HWND,PTR],self.hwnd,handler),'CreateCoreWebView2Controller')

    def _controller_ready(self,result,controller):
        self._emit({'type':'host-progress','stage':'controller-callback','hresult':result})
        check(result,'CreateCoreWebView2Controller')
        self.controller=controller
        com(controller,1,W.ULONG,[])
        web=PTR()
        check(com(controller,25,HRESULT,[C.POINTER(PTR)],C.byref(web)),'get_CoreWebView2')
        self.web=web.value
        self._resize()
        check(com(controller,4,HRESULT,[C.c_int],1),'Controller.Visible')
        nav=self._handler('d33a35bf-1c49-4f98-93ab-006e0533fe1c',[PTR,PTR],self._navigation_completed)
        message=self._handler('57213f19-00e6-49fa-8e07-898ea01ecbd2',[PTR,PTR],self._message_received)
        self.nav_token=C.c_int64()
        self.message_token=C.c_int64()
        check(com(self.web,15,HRESULT,[PTR,C.POINTER(C.c_int64)],nav,C.byref(self.nav_token)),'NavigationCompleted subscription')
        check(com(self.web,34,HRESULT,[PTR,C.POINTER(C.c_int64)],message,C.byref(self.message_token)),'WebMessageReceived subscription')
        if self.html_path:
            check(com(self.web,5,HRESULT,[W.LPCWSTR],self.html_path.as_uri()),'Navigate')
        else:
            check(com(self.web,6,HRESULT,[W.LPCWSTR],'<html><body style="background:#09121e;color:white"><h1 id="title">TouchSign</h1><button id="probe" onclick="document.getElementById(\'title\').textContent=\'Clicked\'">Test</button></body></html>'),'NavigateToString')

    def _navigation_completed(self,sender,args):
        success=C.c_int()
        check(com(args,3,HRESULT,[C.POINTER(C.c_int)],C.byref(success)),'Navigation success')
        if not success.value:
            error=C.c_int()
            com(args,4,HRESULT,[C.POINTER(C.c_int)],C.byref(error))
            raise OSError(f'WebView navigation failed: {error.value}')
        self.ready=True
        self._emit({'type':'host-ready','engine':'WebView2/Win32'})
        if self.on_ready: self.on_ready(self)
        if self.script: self.execute(self.script)

    def _message_received(self,sender,args):
        value=PTR()
        check(com(args,4,HRESULT,[C.POINTER(PTR)],C.byref(value)),'webMessageAsJson')
        try: data=json.loads(C.wstring_at(value))
        finally: self.ole.CoTaskMemFree(value)
        if data.get('type')=='host-test-result':
            self._capture_result(data)
        else:
            self._emit(data)

    def _capture_result(self,data):
        if self.finished: return
        self.finished=True
        data.update(native_window_created=bool(self.hwnd),webview2_initialized=bool(self.web),native_api='ctypes/Win32')
        self.report=data
        def capture_done(result,text):
            try:
                check(result,'Page.captureScreenshot')
                result_data=json.loads(text)
                if self.report_path:
                    path=self.report_path.with_suffix('.png')
                    path.write_bytes(base64.b64decode(result_data['data']))
                    data['screenshot']=str(path)
            except Exception:
                data['capture_error']=traceback.format_exc()
            if self.report_path:
                self.report_path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            self._emit(data)
            if data.get('test_exit_button'):
                self.post({'type':'host-captured'})
            else:
                self.close()
        handler=self._handler('5c4889f0-5ef6-4c5a-952c-d8f1b92d0574',[HRESULT,W.LPCWSTR],capture_done)
        check(com(self.web,36,HRESULT,[W.LPCWSTR,W.LPCWSTR,PTR],'Page.captureScreenshot','{"format":"png"}',handler),'CaptureScreenshot')

    def _timer(self):
        if (not self.ready or self.script) and not self.finished and time.monotonic()>self.deadline:
            self._fail('WebView2 startup or GUI test timed out')
        for _ in range(100):
            try: name,value=self.commands.get_nowait()
            except queue.Empty: break
            if name=='close':
                self.user.DestroyWindow(self.hwnd)
                return
            if not self.ready:
                self.commands.put((name,value))
                break
            if name=='post':
                check(com(self.web,32,HRESULT,[W.LPCWSTR],json.dumps(value,ensure_ascii=False)),'PostWebMessageAsJson')
            elif name=='execute': self.execute(*value)

    def run(self):
        self._emit({'type':'host-progress','stage':'starting'})
        self.thread_id=threading.get_ident()
        self.user=C.WinDLL('user32',use_last_error=True)
        kernel=C.WinDLL('kernel32',use_last_error=True)
        self.ole=C.WinDLL('ole32')
        self.ole.CoInitializeEx.argtypes=[PTR,W.DWORD]
        self.ole.CoInitializeEx.restype=HRESULT
        self.ole.CoTaskMemFree.argtypes=[PTR]
        self.ole.CoTaskMemFree.restype=None
        self.ole.CoUninitialize.argtypes=[]
        self.ole.CoUninitialize.restype=None
        check(self.ole.CoInitializeEx(None,2),'CoInitializeEx STA')
        self._emit({'type':'host-progress','stage':'com-initialized'})
        kernel.GetModuleHandleW.argtypes=[W.LPCWSTR];kernel.GetModuleHandleW.restype=W.HMODULE
        self.user.DefWindowProcW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM];self.user.DefWindowProcW.restype=LRESULT
        self.user.CreateWindowExW.argtypes=[W.DWORD,W.LPCWSTR,W.LPCWSTR,W.DWORD,C.c_int,C.c_int,C.c_int,C.c_int,W.HWND,W.HMENU,W.HINSTANCE,PTR]
        self.user.CreateWindowExW.restype=W.HWND
        self.user.RegisterClassExW.argtypes=[C.POINTER(WNDCLASSEX)];self.user.RegisterClassExW.restype=W.ATOM
        self.user.GetClientRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
        self.user.DestroyWindow.argtypes=[W.HWND]
        self.user.ShowWindow.argtypes=[W.HWND,C.c_int]
        self.user.SetTimer.argtypes=[W.HWND,C.c_size_t,W.UINT,PTR];self.user.SetTimer.restype=C.c_size_t
        self.user.MessageBoxW.argtypes=[W.HWND,W.LPCWSTR,W.LPCWSTR,W.UINT]
        self.user.GetMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT]
        self.user.GetMessageW.restype=C.c_int
        self.user.TranslateMessage.argtypes=[C.POINTER(W.MSG)]
        self.user.DispatchMessageW.argtypes=[C.POINTER(W.MSG)]
        self.user.DispatchMessageW.restype=LRESULT
        def window_proc(hwnd,message,wparam,lparam):
            try:
                if message==5:
                    self._resize();return 0
                if message==0x113:
                    self._timer();return 0
                if message==0x10:
                    self.user.DestroyWindow(hwnd);return 0
                if message==2:
                    self.user.PostQuitMessage(0);return 0
            except Exception: self._fail(traceback.format_exc());return 0
            return self.user.DefWindowProcW(hwnd,message,wparam,lparam)
        self.wndproc=WNDPROC(window_proc)
        instance=kernel.GetModuleHandleW(None)
        class_name='TouchSignWebView_'+str(os.getpid())+'_'+uuid.uuid4().hex
        wc=WNDCLASSEX(C.sizeof(WNDCLASSEX),0,self.wndproc,0,0,instance,None,None,None,None,class_name,None)
        if not self.user.RegisterClassExW(C.byref(wc)): raise C.WinError(C.get_last_error())
        self.hwnd=self.user.CreateWindowExW(0,class_name,'触象 · TouchSign',0x00cf0000,100,60,1480,1000,None,None,instance,None)
        self._emit({'type':'host-progress','stage':'window-created'})
        if not self.hwnd: raise C.WinError(C.get_last_error())
        self.user.ShowWindow(self.hwnd,5)
        self.user.SetTimer(self.hwnd,1,30,None)
        self.profile.mkdir(parents=True,exist_ok=True)
        self.loader=C.WinDLL(str(self.loader_path))
        self._emit({'type':'host-progress','stage':'loader-loaded'})
        create=self.loader.CreateCoreWebView2EnvironmentWithOptions
        create.argtypes=[W.LPCWSTR,W.LPCWSTR,PTR,PTR];create.restype=HRESULT
        handler=self._handler('4e8a3389-c9d8-4bd2-b6b5-124fee6cc14d',[HRESULT,PTR],self._environment_ready)
        check(create(None,str(self.profile),None,handler),'CreateCoreWebView2EnvironmentWithOptions')
        self._emit({'type':'host-progress','stage':'environment-requested'})
        if self.on_created: self.on_created(self)
        message=W.MSG()
        try:
            while self.user.GetMessageW(C.byref(message),None,0,0)>0:
                self.user.TranslateMessage(C.byref(message))
                self.user.DispatchMessageW(C.byref(message))
        finally:
            if self.controller:
                com(self.controller,24,HRESULT,[])
            for pointer in (self.web,self.controller,self.environment):
                if pointer: com(pointer,2,W.ULONG,[])
            self.ole.CoUninitialize()
            self._emit({'type':'host-closed'})
        return self.report


def run_window(html_path=None,profile=None,on_message=None,on_ready=None,selftest_script=None,report=None,loader_path=None,on_created=None):
    return Win32WebView(html_path,profile,on_message,on_ready,selftest_script,report,loader_path,on_created).run()
