# chatbot_wav2lip_server.py - Press SPACE to talk, uses Wav2Lip HTTP server

import requests
import json
import time
import os
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
import torch
from f5_tts.api import F5TTS
import keyboard

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
VOICE_SAMPLE = "voices/hoan_talking_0.wav"
OUTPUT_DIR = "outputs"

# Wav2Lip SERVER settings
WAV2LIP_SERVER_URL = "http://localhost:8000"
AVATAR_PATH = "avatars/hoan_10s.mp4"
WAV2LIP_FPS = 15

# SYSTEM PROMPT
SYSTEM_PROMPT = """You are an AI helping write dialogue for a fictional character in a tech thriller. 
The character is a charismatic but morally ambiguous tech CEO - confident, visionary, and somewhat ruthless in business.
Respond as this character would: ambitious, strategic, occasionally dismissive of regulations, focused on "disruption" and "the future."
Keep responses to 2-3 sentences. Make them sound like a TED talk meets corporate villain monologue."""

# Audio settings
SAMPLE_RATE = 16000
CHUNK = 512
MAX_RECORDING_DURATION = 30
MAX_HISTORY = 10

# ============================================
# INITIALIZE
# ============================================

print("="*60)
print("🎬 INITIALIZING AI AVATAR CHATBOT (Wav2Lip Server)")
print("="*60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ Voice sample not found: {VOICE_SAMPLE}")
    exit(1)

if not os.path.exists(AVATAR_PATH):
    print(f"\n⚠️  Avatar not found: {AVATAR_PATH}")
    AVATAR_PATH = None

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")
if AVATAR_PATH:
    print(f"✅ Avatar: {AVATAR_PATH}")

# Check if Wav2Lip server is running
try:
    response = requests.get(f"{WAV2LIP_SERVER_URL}/health", timeout=2)
    if response.status_code == 200:
        print(f"✅ Wav2Lip server connected: {WAV2LIP_SERVER_URL}")
    else:
        print(f"⚠️  Wav2Lip server not responding properly")
except:
    print(f"❌ Wav2Lip server not running at {WAV2LIP_SERVER_URL}")
    print(f"   Start it first: python wav2lip_server.py")
    exit(1)

# Load Faster-Whisper
print("\n⏳ Loading Whisper...")
start = time.time()
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print(f"✅ Whisper loaded in {time.time() - start:.1f}s")

# Load F5-TTS
print("\n⏳ Loading F5-TTS...")
start = time.time()
tts = F5TTS()
print(f"✅ F5-TTS loaded in {time.time() - start:.1f}s")

# ============================================
# FUNCTIONS
# ============================================

def record_audio_press_to_talk():
    """Record audio while SPACE is held down"""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("\n🎤 Press and HOLD SPACE to record...")
    keyboard.wait('space')
    
    frames = []
    print("🔴 RECORDING... (Release SPACE to stop)")
    
    start_time = time.time()
    max_chunks = int(MAX_RECORDING_DURATION * SAMPLE_RATE / CHUNK)
    chunks_recorded = 0
    
    while keyboard.is_pressed('space') and chunks_recorded < max_chunks:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            chunks_recorded += 1
            
            if chunks_recorded >= max_chunks:
                print(f"\n⏸️  Max duration ({MAX_RECORDING_DURATION}s) reached")
                break
        except Exception as e:
            print(f"⚠️  Audio error: {e}")
            break
    
    duration = time.time() - start_time
    print(f"✅ Recording stopped ({duration:.1f}s)")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames

def save_audio(frames, filename):
    """Save audio frames to file"""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    p.terminate()
    
    return filename

def transcribe_audio(audio_file):
    """Transcribe audio with Faster-Whisper"""
    print(f"⏳ Transcribing...", end='', flush=True)
    start = time.time()
    
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    
    elapsed = time.time() - start
    print(f"\r✅ Transcribed in {elapsed:.1f}s")
    
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

def generate_wav2lip_video(audio_path, output_video_name):
    """Generate video with Wav2Lip SERVER (models stay loaded!)"""
    
    if not AVATAR_PATH:
        return None
    
    print(f"\n🎬 Generating video with Wav2Lip...")
    start_time = time.time()
    
    try:
        # Get absolute paths
        abs_audio_path = os.path.abspath(audio_path)
        abs_avatar_path = os.path.abspath(AVATAR_PATH)
        abs_output_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_video_name))
        
        # Send HTTP request to Wav2Lip server
        response = requests.post(
            f"{WAV2LIP_SERVER_URL}/generate",
            json={
                "audio_path": abs_audio_path,
                "avatar_path": abs_avatar_path,
                "output_path": abs_output_path,
                "fps": WAV2LIP_FPS
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            print(f"✅ Video saved: {abs_output_path}")
            print(f"⏱️  Wav2Lip took {elapsed:.1f}s")
            return abs_output_path
        else:
            print(f"❌ Wav2Lip server error: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except requests.Timeout:
        print(f"❌ Wav2Lip server timeout")
        return None
    except Exception as e:
        print(f"❌ Wav2Lip error: {e}")
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

def play_video(video_path):
    """Play video file"""
    if not os.path.exists(video_path):
        print(f"⚠️  Video not found: {video_path}")
        return
    
    try:
        os.startfile(video_path)
        print(f"🎥 Playing video: {video_path}")
    except Exception as e:
        print(f"⚠️  Could not play video: {e}")
        print(f"📁 Video saved: {video_path}")

# ============================================
# MAIN LOOP
# ============================================

def run_press_to_talk_chatbot():
    """Press-to-talk chatbot with Wav2Lip video"""
    
    print("\n" + "="*60)
    print("🎬 AI AVATAR CHATBOT - Ready!")
    print("="*60)
    print(f"Voice: {VOICE_SAMPLE}")
    if AVATAR_PATH:
        print(f"Avatar: {AVATAR_PATH}")
    print(f"LLM: {OLLAMA_MODEL}")
    print(f"Lip-sync: Wav2Lip Server (models cached!)")
    print("\n📢 Controls:")
    print("   • Press and HOLD SPACE to record")
    print("   • Release SPACE to stop and process")
    print("   • Press ESC or say 'goodbye' to exit")
    print("="*60)
    
    conversation_history = []
    turn = 0
    
    try:
        while True:
            # Record audio
            frames = record_audio_press_to_talk()
            
            if not frames or len(frames) < 5:
                print("⚠️  Recording too short, try again")
                continue
            
            turn += 1
            start_total = time.time()
            
            # Save audio
            audio_file = os.path.join(OUTPUT_DIR, f"recording_{turn:03d}.wav")
            save_audio(frames, audio_file)
            
            # Transcribe
            user_input = transcribe_audio(audio_file)
            print(f"\n💬 You: \"{user_input}\"")
            
            # Check for exit
            if any(word in user_input.lower() for word in ['goodbye', 'exit', 'quit']):
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                print("⚠️  No speech detected")
                turn -= 1
                continue
            
            # Get LLM response
            response_text = get_llm_response(user_input, conversation_history)
            
            # Generate voice
            audio_filename = f"response_{turn:03d}.wav"
            audio_path = generate_voice(response_text, audio_filename)
            
            if not audio_path:
                continue
            
            # Generate video with Wav2Lip SERVER
            if AVATAR_PATH:
                video_filename = f"response_{turn:03d}.mp4"
                video_path = generate_wav2lip_video(audio_path, video_filename)
                
                total_time = time.time() - start_total
                print(f"⏱️  Total: {total_time:.1f}s")
                
                if video_path:
                    play_video(video_path)
                else:
                    print("⚠️  Video failed, playing audio only")
                    play_audio(audio_path)
            else:
                total_time = time.time() - start_total
                print(f"\n⏱️  Total: {total_time:.1f}s")
                play_audio(audio_path)
            
            # Update history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response_text}")
            
            print("\n" + "-"*60)
            
            # Check for ESC
            if keyboard.is_pressed('esc'):
                print(f"\n👋 Goodbye! Had {turn} conversations.")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Goodbye! Had {turn} conversations.")

if __name__ == "__main__":
    run_press_to_talk_chatbot()