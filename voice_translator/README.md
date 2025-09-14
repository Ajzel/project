Voice-to-Voice AI Translator

A Python Flask web application that provides real-time voice translation between multiple languages.

## Features

- 🎤 Voice-to-text using speech recognition
- 🌍 Text translation between 11 languages
- 🔊 Text-to-speech for translated content
- 💻 Manual text input option
- 🔄 Language swapping functionality
- 📱 Responsive web interface

## Supported Languages

- English, Spanish, French, German, Italian
- Portuguese, Russian, Japanese, Korean
- Chinese (Mandarin), Malayalam

## Installation

1. **Clone or create the project directory:**
   ```bash
   mkdir voice_translator
   cd voice_translator
   ```

2. **Create the folder structure:**
   ```
   voice_translator/
   ├── app.py
   ├── requirements.txt
   ├── templates/
   │   └── index.html
   ├── static/
   │   ├── css/
   │   │   └── style.css
   │   └── js/
   │       └── script.js
   └── README.md
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install portaudio19-dev python3-pyaudio espeak espeak-data
   ```

5. **For Windows users:**
   - Download and install PyAudio from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Install espeak: `choco install espeak` (if you have Chocolatey)

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Using the application:**
   - Select source and target languages
   - Click the microphone button to start voice recording
   - Or type text manually in the text area
   - Listen to the translation using the Play button
   - Use the swap button to quickly switch languages

## API Endpoints

- `GET /` - Main web interface
- `POST /translate` - Translate text between languages
- `POST /speech-to-text` - Convert speech to text
- `POST /text-to-speech` - Convert text to speech
- `GET /get-languages` - Get available languages

## Troubleshooting

### Microphone Issues
- Ensure microphone permissions are granted
- Check if microphone is working in other applications
- Try running as administrator (Windows) or with sudo (Linux)

### Audio Output Issues
- Make sure speakers/headphones are connected
- Check system audio settings
- Verify TTS engine is properly installed

### Translation Issues
- Check internet connection (required for Google Translate)
- Try the fallback MyMemory API if Google Translate fails

## Dependencies

- **Flask**: Web framework
- **SpeechRecognition**: Speech-to-text conversion  
- **pyttsx3**: Text-to-speech synthesis
- **googletrans**: Google Translate API
- **requests**: HTTP requests for translation APIs
- **pyaudio**: Audio input/output

## Notes

- Requires internet connection for translation services
- Microphone access required for voice input
- TTS quality depends on system voice engines
- Some languages may have limited TTS support

## License

MIT License - Feel free to modify and distribute as needed.
"""