# chatbot_voice_interactive.py - Full Voice-to-Voice Chatbot

import requests
import json
import time
import os
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
from f5_tts.api import F5TTS

# ============================================
# CONFIGURATION
# ============================================

# Ollama
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Voice
VOICE_SAMPLE = "voices/hoan_talking_0.wav"
OUTPUT_DIR = "outputs"

# Recording settings
RECORD_SECONDS = 5  # How long to record when you speak
SAMPLE_RATE = 16000

# Conversation
MAX_HISTORY = 10

# ============================================
# INITIALIZE
# ============================================

print("="*60)
print("🎙️  INITIALIZING VOICE-TO-VOICE CHATBOT")
print("="*60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check voice sample
if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ Voice sample not found: {VOICE_SAMPLE}")
    exit(1)

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")

# Load Whisper
print("\n⏳ Loading Whisper (speech recognition)...")
start = time.time()
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print(f"✅ Whisper loaded in {time.time() - start:.1f}s")

# Load F5-TTS
print("\n⏳ Loading F5-TTS (voice synthesis)...")
start = time.time()
tts = F5TTS()
print(f"✅ F5-TTS loaded in {time.time() - start:.1f}s")

# ============================================
# FUNCTIONS
# ============================================

def record_audio(duration=RECORD_SECONDS):
    """Record audio from microphone"""
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    
    print(f"\n🎤 Recording for {duration} seconds...")
    print("🗣️  SPEAK NOW!")
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    
    # Record
    for i in range(0, int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("✅ Recording complete")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to file
    filename = os.path.join(OUTPUT_DIR, f"recording_{int(time.time())}.wav")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

def transcribe_audio(audio_file):
    """Transcribe audio with Whisper"""
    
    print(f"⏳ Transcribing...", end='', flush=True)
    start = time.time()
    
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    
    elapsed = time.time() - start
    print(f"\r✅ Transcribed in {elapsed:.1f}s")
    
    return text.strip()

def get_llm_response(user_input, history):
    """Get response from Ollama"""
    
    context = "\n".join(history[-MAX_HISTORY:])
    prompt = f"{context}\nUser: {user_input}\nAssistant:"
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 150,
                }
            },
            timeout=60,
            stream=True
        )
        
        full_response = ""
        print("\n🤖 Avatar: ", end='', flush=True)
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    print(chunk, end='', flush=True)
                    full_response += chunk
        
        print()
        return full_response.strip()
        
    except Exception as e:
        print(f"\n❌ LLM Error: {e}")
        return "Sorry, I encountered an error."

def generate_voice(text, filename):
    """Generate voice with F5-TTS"""
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    print(f"🎙️  Generating voice...", end='', flush=True)
    start = time.time()
    
    try:
        tts.infer(
            ref_file=VOICE_SAMPLE,
            ref_text="",
            gen_text=text,
            file_wave=output_path,
        )
        
        elapsed = time.time() - start
        print(f"\r✅ Voice generated in {elapsed:.1f}s")
        return output_path
        
    except Exception as e:
        print(f"\r❌ Voice generation failed: {e}")
        return None

def play_audio(audio_path):
    """Play audio file"""
    try:
        import subprocess
        subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).wait()
    except:
        print(f"📁 Audio saved: {audio_path}")

# ============================================
# MAIN LOOP
# ============================================

def run_voice_chatbot():
    """Voice-to-voice conversation loop"""
    
    print("\n" + "="*60)
    print("🎙️  VOICE-TO-VOICE CHATBOT - Ready!")
    print("="*60)
    print(f"Voice Model: {VOICE_SAMPLE}")
    print(f"LLM: {OLLAMA_MODEL}")
    print(f"Recording: {RECORD_SECONDS} seconds per turn")
    print("\nControls:")
    print("  - Press ENTER to record your voice")
    print("  - Type 'quit' to exit")
    print("  - Type text to use text mode")
    print("="*60)
    
    conversation_history = []
    turn = 0
    
    while True:
        try:
            print("\n" + "-"*60)
            
            # Get user input
            user_input = input("\n💬 Press ENTER to speak (or type text): ").strip()
            
            # Check for quit
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"\n👋 Goodbye! Had {turn} conversations.")
                break
            
            turn += 1
            start_total = time.time()
            
            # Voice input if Enter pressed
            if not user_input:
                # Record audio
                audio_file = record_audio(RECORD_SECONDS)
                
                # Transcribe
                user_input = transcribe_audio(audio_file)
                print(f"\n💬 You said: \"{user_input}\"")
                
                if not user_input:
                    print("⚠️  No speech detected, try again")
                    turn -= 1
                    continue
            
            # Get LLM response
            print("\n🤔 Thinking...", end='', flush=True)
            start_llm = time.time()
            
            response_text = get_llm_response(user_input, conversation_history)
            llm_time = time.time() - start_llm
            
            # Generate voice
            audio_filename = f"response_{turn:03d}_{int(time.time())}.wav"
            audio_path = generate_voice(response_text, audio_filename)
            
            if audio_path:
                total_time = time.time() - start_total
                tts_time = total_time - llm_time
                
                print(f"\n⏱️  Total: {total_time:.1f}s (Transcribe+LLM: {llm_time:.1f}s + TTS: {tts_time:.1f}s)")
                
                # Play response
                print("\n🔊 Playing response...")
                play_audio(audio_path)
            
            # Update history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response_text}")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye! Had {turn} conversations.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    run_voice_chatbot()