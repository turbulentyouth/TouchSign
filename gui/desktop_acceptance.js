(async()=>{
 const $=id=>document.getElementById(id);
 const sleep=ms=>new Promise(r=>setTimeout(r,ms));
 const wait=async(fn,label,ms=10000)=>{const start=Date.now();while(!fn()){if(Date.now()-start>ms)throw Error('Timeout: '+label);await sleep(80);}};
 const steps=[];
 const click=async(id,done,label)=>{if($(id).disabled)throw Error('Disabled: '+id);$(id).click();await wait(done,label);steps.push(label);};
 try{
  await wait(()=>window.touchsign?.ready && window.touchsign.state?.phase,'bridge ready');
  if(!$('robot-settings').hidden)throw Error('Settings should start hidden');
  if($('detect').parentElement!==$('exit').parentElement)throw Error('Toolbar layout');
  await click('toggle-settings',()=>!$('robot-settings').hidden,'expand settings');
  $('speech-toggle').click();await sleep(150);
  await click('robot-connect',()=>window.touchsign.state.connected&&!window.touchsign.state.busy,'mock connect');
  await click('teach-start',()=>window.touchsign.state.teaching&&!window.touchsign.state.busy,'gravity teaching');
  $('angles').value='[0.1,0.1,0.1,0.1,0.1,0.1,0.1]';$('mock-drag').click();await sleep(350);
  await click('save-detect',()=>!!window.touchsign.state.points?.detect&&!window.touchsign.state.busy,'save detect point');
  $('angles').value='[0,0,0,0,0,0,0]';$('mock-drag').click();await sleep(350);
  await click('save-home',()=>!!window.touchsign.state.points?.home&&!window.touchsign.state.busy,'save home point');
  await click('teach-end',()=>!window.touchsign.state.teaching&&!window.touchsign.state.busy,'end teaching');
  await click('toggle-settings',()=>$('robot-settings').hidden,'collapse settings');
  await click('detect',()=>window.touchsign.state.phase==='complete','one detection cycle',15000);
  if(!$('result-kind').textContent.includes('模拟')||$('result-glyph').textContent!=='鼠')throw Error('Result not rendered');
  await sleep(500);
  if(window.touchsign.state.completed!==1)throw Error('Cycle did not remain single');
  if(!$('error').hidden)throw Error('UI error: '+$('error').textContent);
  const rect=$('exit').getBoundingClientRect();if(rect.width<40||rect.top<0)throw Error('Exit not visible');
  chrome.webview.addEventListener('message',e=>{if(e.data.type==='host-captured')$('exit').click();});
  chrome.webview.postMessage({type:'host-test-result',passed:true,test_exit_button:true,steps,completed:window.touchsign.state.completed,result:$('result-title').textContent,settings_collapsed:$('robot-settings').hidden,toolbar_exit_visible:true});
 }catch(e){chrome.webview.postMessage({type:'host-test-result',passed:false,error:String(e),steps,state:window.touchsign?.state,ui_error:$('error')?.textContent});}
})();
