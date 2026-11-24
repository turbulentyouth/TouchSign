(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const keys = ['rat','ox','tiger','rabbit','dragon','snake','horse','goat','monkey','rooster','dog','pig'];
  const glyphs = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪'];
  let speech = true;
  let difference = false;
  let settingsLoaded = false;
  let state = {};
  const history = [];
  window.touchsign = {state, events: [], ready: false};
  function error(message) {$('error').textContent=message; $('error').hidden=false;}
  function send(name, extra={}) {
    $('error').hidden=true;
    if (!window.chrome?.webview) {error('内嵌窗口消息桥未连接'); return;}
    chrome.webview.postMessage({type:'command', name, ...extra});
  }
  function click(id, name, args=()=>({})) {$(id).onclick=()=>{try{send(name,args());}catch(e){error(e.message);}};}
  function renderStatus(s) {
    if (!settingsLoaded && s.settings) {
      const v=s.settings; $('channel').value=v.channel; $('speed').value=v.speed_percent;
      $('open-width').value=v.open_width*1000; $('close-width').value=v.close_width*1000; $('force').value=v.gripper_force; $('close-speed').value=v.gripper_close_speed*1000;
      settingsLoaded=true;
    }
    state=s; window.touchsign.state=s;
    const connected=!!s.connected, busy=!!s.busy, teach=!!s.teaching;
    $('system-status').textContent=({disconnected:'未连接',disabled:'等待使能',idle:'待检测',teaching:'示教中',switching:'切换模式',preparing:'准备检测',approaching:'到达检测点',gripping:'等待夹爪到位',recognizing:'识别中',releasing:'释放中',returning:'收回中',complete:'本轮完成',stopped:'已停止',fault:'故障'})[s.phase] || '就绪';
    $('robot-status').textContent=s.message || '就绪';
    $('model-status').textContent=s.model || '';
    $('sensor-status').textContent=s.sensor ? '传感器已连接' : 'NO INPUT';
    if (!s.sensor) {
      $('sensor-frame').hidden=true; $('sensor-frame').style.setProperty('display','none','important');
      $('sensor-placeholder').hidden=false; $('sensor-placeholder').style.setProperty('display','block','important');
    }
    $('contact-status').textContent=s.calibrated ? '空载基线已校准' : '请移开所有物体，再进行空载校准';
    const points=s.points || {};
    $('points').textContent=`检测点：${points.detect?'已保存':'未保存'} · 收回点：${points.home?'已保存':'未保存'}`;
    $('detect').disabled=!connected||busy||teach||!['idle','complete'].includes(s.phase)||!points.detect||!points.home;
    $('stop').disabled=!connected||!!s.finishing||teach||!points.home||['disabled','switching'].includes(s.phase);
    $('robot-connect').disabled=connected||busy;
    $('robot-disconnect').disabled=!connected;
    $('backend').disabled=connected; $('channel').disabled=connected;
    $('teach-start').disabled=!connected||busy||teach||!['idle','disabled','complete'].includes(s.phase);
    for(const id of ['save-detect','save-home','teach-end']) $(id).disabled=!teach||busy;
    $('mock-drag').disabled=!teach||busy||s.backend==='nero';
    for(const id of ['calibrate','scan','camera-connect','settings-save','speed','open-width','close-width','close-speed','force']) $(id).disabled=busy||teach;
  }
  function receive(e) {
    const m=e.data; if (!m || typeof m!=='object') return;
    window.touchsign.events.push(m.type); if(window.touchsign.events.length>100)window.touchsign.events.shift();
    if(m.type==='status') renderStatus(m);
    if(m.type==='error') error(m.message);
    if(m.type==='frame') { $('sensor-frame').src='data:image/jpeg;base64,'+m.jpeg; $('sensor-frame').hidden=false; $('sensor-frame').style.setProperty('display','block','important'); $('sensor-placeholder').hidden=true; $('sensor-placeholder').style.setProperty('display','none','important'); }
    if(m.type==='cameras') { $('camera').replaceChildren(); for(const c of m.items||[]) {const o=new Option(c.name||c.label||('摄像头 '+c.index),c.index); $('camera').add(o);} }
    if(m.type==='result') {
      $('result-glyph').textContent=m.glyph||'—'; $('result-title').textContent=m.title||'未识别';
      $('result-kind').textContent=m.simulated?'离线模拟 · 非真实识别':'本轮检测结果';
      $('result-description').textContent=m.simulated?'模拟流程结果，不写入真实识别记录':(m.message||'识别结束，松开后收回');
      document.querySelectorAll('.zodiac-native').forEach(x=>x.classList.toggle('active',x.dataset.key===m.key));
      history.unshift(`${new Date().toLocaleTimeString()}　${m.simulated?'[模拟] ':''}${m.title||'未识别'}`); history.splice(4);
      $('history').textContent=history.join('\n');
    }
  }
  keys.forEach((key,i)=> { $('mock-key').add(new Option(glyphs[i],key)); const box=document.createElement('div'); box.className='zodiac-native'; box.dataset.key=key; const g=document.createElement('strong'); g.textContent=glyphs[i]; box.append(g,document.createTextNode(key.toUpperCase())); $('zodiac-list').append(box); });
  $('mock-key').add(new Option('模拟识别超时','timeout'));
  $('toggle-settings').onclick=()=>{const hide=!$('robot-settings').hidden; $('robot-settings').hidden=hide; $('toggle-settings').textContent=hide?'展开机械臂设置':'收起机械臂设置'; $('toggle-settings').setAttribute('aria-expanded',String(!hide));};
  $('backend').onchange=()=>{$('real-confirm-label').hidden=$('backend').value!=='nero'; $('mock-controls').hidden=$('backend').value!=='mock';};
  click('robot-connect','connect',()=>({backend:$('backend').value,channel:$('channel').value,real_confirmed:$('real-confirm').checked}));
  click('robot-disconnect','disconnect'); click('teach-start','teach_start'); click('teach-end','teach_end');
  click('save-detect','save_point',()=>({point:'detect'})); click('save-home','save_point',()=>({point:'home'}));
  click('mock-drag','mock_drag',()=>({joints:JSON.parse($('angles').value)}));
  click('detect','detect',()=>({key:$('mock-key').value})); click('stop','stop'); click('exit','close');
  click('scan','scan_cameras'); click('camera-connect','connect_camera',()=>({index:Number($('camera').value)})); click('calibrate','calibrate');
  const preview=document.createElement('button');preview.className='runtime-button';preview.textContent='原始图 · 切换差分';preview.onclick=()=>{difference=!difference;send('preview',{difference});preview.textContent=difference?'差分图 · 切换原图':'原始图 · 切换差分';};$('toolbar').insertBefore(preview,$('speech-toggle'));
  const reload=document.createElement('button');reload.className='runtime-button';reload.textContent='重新加载模型';reload.onclick=()=>send('reload_model');$('toolbar').insertBefore(reload,$('model-status'));
  const disconnect=document.createElement('button');disconnect.className='runtime-button';disconnect.textContent='断开传感器';disconnect.onclick=()=>send('disconnect_camera');$('camera-connect').after(disconnect);
  click('settings-save','settings',()=>({values:{speed_percent:Number($('speed').value),open_width:Number($('open-width').value)/1000,close_width:Number($('close-width').value)/1000,gripper_close_speed:Number($('close-speed').value)/1000,gripper_force:Number($('force').value)}}));
  click('speech-test','test_speech',()=>({key:$('mock-key').value}));
  $('speech-toggle').onclick=()=>{speech=!speech;send('speech',{enabled:speech});$('speech-toggle').textContent='语音播报 · '+(speech?'开':'关');};
  function clock(){const d=new Date();$('clock').textContent=d.toLocaleTimeString();$('date').textContent=d.toLocaleDateString();} clock();setInterval(clock,1000);
  if(window.chrome?.webview) {chrome.webview.addEventListener('message',receive); window.touchsign.ready=true;send('ready');} else error('请从 TouchSign 桌面应用启动此页面');
})();
