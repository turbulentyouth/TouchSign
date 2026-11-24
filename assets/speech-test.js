// A new recognition immediately replaces the current clip.
function stopRecognitionSpeech(audio) {
    audio._pendingSpeech = null;
    if (audio.dataset.testing) return;
    audio._speechToken = (audio._speechToken || 0) + 1;
    audio.onended = audio.onerror = null;
    audio._speechBusy = false;
    audio.pause();
}

function queueRecognitionSpeech(audio, source) {
    if (audio.dataset.testing) return;
    stopRecognitionSpeech(audio);
    const token = audio._speechToken;
    audio._speechBusy = true;
    function finish() {
        if (token !== audio._speechToken) return;
        audio._speechBusy = false;
        audio.onended = audio.onerror = null;
    }
    function failed() {
        if (token !== audio._speechToken) return;
        finish();
        ++audio._speechToken;
        Quasar.Notify.create('语音播放失败，请检查浏览器声音权限');
    }
    audio.onended = finish;
    audio.onerror = failed;
    audio.src = source;
    audio.play().catch(failed);
}

function toggleSpeechTest(button, audio, playlist) {
    const label = button.querySelector('span');
    if (audio.dataset.testing) {
        audio._finishTest();
        return;
    }
    stopRecognitionSpeech(audio);
    audio.dataset.testing = 'true';
    let index = 0;
    function announce(key) {
        audio.dispatchEvent(new CustomEvent('speech-test', {detail: key}));
    }
    function finish() {
        audio.pause();
        audio.onended = audio.onerror = null;
        delete audio.dataset.testing;
        label.textContent = '测试声音';
        announce(null);
    }
    audio._finishTest = finish;
    function failed() {
        if (audio.onerror !== failed) return;
        finish();
        Quasar.Notify.create('试听失败，请检查声音输出后重试');
    }
    function playNext() {
        if (index === playlist.length) {
            finish();
            return;
        }
        const [key, glyph] = playlist[index++];
        label.textContent = '停止试听 · ' + glyph;
        announce(key);
        audio.src = '/assets/speech/' + key + '.wav';
        audio.play().catch(failed);
    }
    audio.onerror = failed;
    audio.onended = playNext;
    playNext();
}
