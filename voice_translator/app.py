from flask import Flask, render_template, request, jsonify
import requests
import json
import speech_recognition as sr
import threading
import time
from deep_translator import GoogleTranslator, MyMemoryTranslator

app = Flask(__name__)

# Initialize components
recognizer = sr.Recognizer()

# Language mappings
LANGUAGE_CODES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ml': 'Malayalam'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        from_lang = data.get('from_lang', 'en')
        to_lang = data.get('to_lang', 'es')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Use deep-translator with Google Translate
        try:
            translator = GoogleTranslator(source=from_lang, target=to_lang)
            translated_text = translator.translate(text)
        except Exception as google_error:
            print(f"Google Translate error: {google_error}")
            # Fallback to MyMemory translator
            try:
                translator = MyMemoryTranslator(source=from_lang, target=to_lang)
                translated_text = translator.translate(text)
            except Exception as mymemory_error:
                print(f"MyMemory error: {mymemory_error}")
                # Final fallback to direct API call
                try:
                    url = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(text)}&langpair={from_lang}|{to_lang}"
                    response = requests.get(url, timeout=10)
                    response_data = response.json()
                    
                    if response_data.get('responseStatus') == 200:
                        translated_text = response_data['responseData']['translatedText']
                    else:
                        return jsonify({'error': 'All translation services failed'}), 500
                except Exception as api_error:
                    print(f"API error: {api_error}")
                    return jsonify({'error': 'Translation failed - all services unavailable'}), 500
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translated_text': translated_text,
            'from_lang': from_lang,
            'to_lang': to_lang
        })
        
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert speech from microphone to text"""
    try:
        data = request.get_json()
        language = data.get('language', 'en-US')
        
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio, language=language)
        
        return jsonify({
            'success': True,
            'text': text,
            'language': language
        })
        
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {e}'}), 500
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.WaitTimeoutError:
        return jsonify({'error': 'Listening timeout'}), 408
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Text-to-speech is now handled by browser - this endpoint for compatibility"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        return jsonify({
            'success': True,
            'message': 'Use browser speech synthesis',
            'text': text,
            'language': language
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-languages')
def get_languages():
    """Get available languages"""
    return jsonify(LANGUAGE_CODES)

if __name__ == '__main__':
    print("Starting Voice Translator Server...")
    print("Dependencies required: Flask, speechrecognition, deep-translator, requests")
    print("Optional: pyaudio (for microphone input)")
    app.run(debug=True, host='0.0.0.0', port=5000)