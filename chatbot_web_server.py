# chatbot_web_server.py - Flask server for voice chatbot

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests
import json
import time
import os
import tempfile
from faster_whisper import WhisperModel
from f5_tts.api import F5TTS
import base64

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
VOICE_SAMPLE = "voices/hoan_talking_0.wav"
OUTPUT_DIR = "outputs"

# SYSTEM PROMPT
SYSTEM_PROMPT = """You are a friendly, helpful AI assistant with a warm and conversational personality. 
Speak naturally and casually, as if chatting with a good friend. 
Keep your responses concise and engaging - aim for 2-3 sentences unless the user specifically asks for more detail.
Be enthusiastic, supportive, and personable in your tone."""

MAX_HISTORY = 10

# ============================================
# INITIALIZE FLASK
# ============================================

app = Flask(__name__)
CORS(app)  # Enable CORS for external access

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store conversation history (in production, use sessions or database)
conversation_histories = {}

print("="*60)
print("🎙️  INITIALIZING WEB CHATBOT SERVER")
print("="*60)

if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ Voice sample not found: {VOICE_SAMPLE}")
    exit(1)

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")

# Load Faster-Whisper
print("\n⏳ Loading Faster-Whisper...")
start = time.time()
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print(f"✅ Faster-Whisper loaded in {time.time() - start:.1f}s")

# Load F5-TTS
print("\n⏳ Loading F5-TTS...")
start = time.time()
tts = F5TTS()
print(f"✅ F5-TTS loaded in {time.time() - start:.1f}s")

print("\n🌐 Server ready!")

# ============================================
# HELPER FUNCTIONS
# ============================================

def transcribe_audio(audio_file):
    """Transcribe audio with Faster-Whisper"""
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def get_llm_response(user_input, conversation_history):
    """Get response from Ollama with system prompt"""
    messages = [SYSTEM_PROMPT]
    
    for msg in conversation_history[-MAX_HISTORY:]:
        messages.append(msg)
    
    messages.append(f"User: {user_input}")
    messages.append("Assistant:")
    
    prompt = "\n".join(messages)
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150,
                }
            },
            timeout=60
        )
        
        data = response.json()
        return data.get("response", "").strip()
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return "Sorry, I encountered an error."

def generate_voice(text, filename):
    """Generate voice with F5-TTS"""
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        tts.infer(
            ref_file=VOICE_SAMPLE,
            ref_text="",
            gen_text=text,
            file_wave=output_path,
        )
        return output_path
    except Exception as e:
        print(f"❌ Voice generation failed: {e}")
        return None

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process audio and return response"""
    try:
        # Get session ID (for conversation history)
        session_id = request.form.get('session_id', 'default')
        
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save to temporary file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_audio.name)
        temp_audio.close()
        
        print(f"\n📥 Received audio from session: {session_id}")
        
        # Transcribe
        start = time.time()
        user_input = transcribe_audio(temp_audio.name)
        transcribe_time = time.time() - start
        print(f"✅ Transcribed in {transcribe_time:.1f}s: \"{user_input}\"")
        
        # Clean up temp file
        os.unlink(temp_audio.name)
        
        if not user_input:
            return jsonify({'error': 'No speech detected'}), 400
        
        # Get conversation history for this session
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []
        
        conversation_history = conversation_histories[session_id]
        
        # Get LLM response
        start = time.time()
        response_text = get_llm_response(user_input, conversation_history)
        llm_time = time.time() - start
        print(f"✅ LLM responded in {llm_time:.1f}s")
        
        # Generate voice
        start = time.time()
        audio_filename = f"response_{session_id}_{int(time.time())}.wav"
        audio_path = generate_voice(response_text, audio_filename)
        tts_time = time.time() - start
        print(f"✅ TTS generated in {tts_time:.1f}s")
        
        if not audio_path:
            return jsonify({'error': 'Voice generation failed'}), 500
        
        # Update conversation history
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Assistant: {response_text}")
        
        # Return response
        return jsonify({
            'transcription': user_input,
            'response': response_text,
            'audio_url': f'/api/audio/{audio_filename}',
            'timing': {
                'transcribe': round(transcribe_time, 2),
                'llm': round(llm_time, 2),
                'tts': round(tts_time, 2),
                'total': round(transcribe_time + llm_time + tts_time, 2)
            }
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/<filename>')
def get_audio(filename):
    """Serve audio file"""
    audio_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': 'Audio file not found'}), 404

@app.route('/api/clear/<session_id>', methods=['POST'])
def clear_history(session_id):
    """Clear conversation history for a session"""
    if session_id in conversation_histories:
        conversation_histories[session_id] = []
    return jsonify({'status': 'cleared'})

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Starting Flask server...")
    print("="*60)
    print("\n📍 Local access: http://localhost:5000")
    print("📍 Network access: http://YOUR_IP:5000")
    print("\n⚠️  To allow external access:")
    print("   1. Find your local IP: ipconfig (Windows)")
    print("   2. Share the URL with external users")
    print("   3. Make sure your firewall allows port 5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)