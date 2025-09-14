let isRecording = false;
let currentTranslation = '';

function updateStatus(message, isError = false, isSuccess = false) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = 'status';
    if (isError) status.className += ' error';
    if (isSuccess) status.className += ' success';
}

async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    const micBtn = document.getElementById('micBtn');
    const fromLang = document.getElementById('fromLang').value;
    
    try {
        isRecording = true;
        micBtn.classList.add('recording');
        updateStatus('Listening... Speak now!');
        
        const response = await fetch('/speech-to-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                language: fromLang + '-US'  // Add country code for speech recognition
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('originalContent').textContent = data.text;
            document.getElementById('originalText').style.display = 'block';
            await translateText(data.text);
        } else {
            updateStatus(data.error || 'Speech recognition failed', true);
        }
        
    } catch (error) {
        console.error('Recording error:', error);
        updateStatus('Recording failed: ' + error.message, true);
    } finally {
        stopRecording();
    }
}

function stopRecording() {
    isRecording = false;
    document.getElementById('micBtn').classList.remove('recording');
    if (document.getElementById('status').textContent === 'Listening... Speak now!') {
        updateStatus('Click the microphone to start speaking');
    }
}

async function translateText(text) {
    const fromLang = document.getElementById('fromLang').value;
    const toLang = document.getElementById('toLang').value;
    
    updateStatus('Translating...');
    
    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                from_lang: fromLang,
                to_lang: toLang
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentTranslation = data.translated_text;
            document.getElementById('translatedContent').textContent = currentTranslation;
            document.getElementById('translatedText').style.display = 'block';
            document.getElementById('translatedText').className = 'text-display translated-text';
            updateStatus('Translation complete!', false, true);
        } else {
            throw new Error(data.error || 'Translation failed');
        }
        
    } catch (error) {
        console.error('Translation error:', error);
        document.getElementById('translatedText').className = 'text-display error';
        document.getElementById('translatedContent').textContent = 'Translation failed: ' + error.message;
        document.getElementById('translatedText').style.display = 'block';
        updateStatus('Translation error', true);
    }
}

async function translateManualText() {
    const text = document.getElementById('manualText').value.trim();
    if (!text) {
        updateStatus('Please enter text to translate', true);
        return;
    }
    
    document.getElementById('originalContent').textContent = text;
    document.getElementById('originalText').style.display = 'block';
    await translateText(text);
}

async function speakTranslation() {
    if (!currentTranslation) return;
    
    const toLang = document.getElementById('toLang').value;
    const playBtn = document.getElementById('playBtn');
    
    try {
        playBtn.disabled = true;
        playBtn.textContent = '🔊 Playing...';
        
        const response = await fetch('/text-to-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: currentTranslation,
                language: toLang
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Text-to-speech failed');
        }
        
        // Reset button after a delay (TTS runs in background)
        setTimeout(() => {
            playBtn.disabled = false;
            playBtn.textContent = '🔊 Play';
        }, 2000);
        
    } catch (error) {
        console.error('TTS error:', error);
        playBtn.disabled = false;
        playBtn.textContent = '🔊 Play';
        updateStatus('Speech synthesis error: ' + error.message, true);
    }
}

function swapLanguages() {
    const fromLang = document.getElementById('fromLang');
    const toLang = document.getElementById('toLang');
    
    const fromValue = fromLang.value;
    const toValue = toLang.value;
    
    fromLang.value = toValue;
    toLang.value = fromValue;
    
    // Clear previous results
    document.getElementById('originalText').style.display = 'none';
    document.getElementById('translatedText').style.display = 'none';
    updateStatus('Languages swapped! Ready for translation');
}

// Initialize when page loads
window.onload = function() {
    updateStatus('Ready! Click the microphone to start speaking or type text below');
};

// Test in browser console (F12):
speechSynthesis.getVoices().length  // Should be > 0
speechSynthesis.speak(new SpeechSynthesisUtterance('test'))