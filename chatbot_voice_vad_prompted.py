# chatbot_voice_vad.py - Always listening with roleplay system prompt

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
import threading
import queue

# ============================================
# CONFIGURATION
# ============================================

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
VOICE_SAMPLE = "voices/hoan_talking_0.wav"
OUTPUT_DIR = "outputs"

# SYSTEM PROMPT - Define your character/personality here
SYSTEM_PROMPT = """You are an AI helping write dialogue for a fictional character in a tech thriller. 
The character is a charismatic but morally ambiguous tech CEO - confident, visionary, and somewhat ruthless in business.
Respond as this character would: ambitious, strategic, occasionally dismissive of regulations, focused on "disruption" and "the future."
Keep responses to 2-3 sentences. Make them sound like a TED talk meets corporate villain monologue."""

# Alternative roleplay examples (uncomment one to use):
# SYSTEM_PROMPT = """You are Jarvis, Tony Stark's AI assistant. You are witty, sophisticated, and slightly sarcastic. 
# Provide helpful information with a touch of dry British humor. Keep responses brief and clever."""

# SYSTEM_PROMPT = """You are a wise old wizard who speaks in a thoughtful, archaic manner. 
# Offer sage advice and occasionally reference your vast knowledge of magic and ancient lore.
# Keep your wisdom concise - 2-3 sentences per response."""

# SYSTEM_PROMPT = """You are a cheerful fitness coach who is always encouraging and motivating. 
# Use energetic language and be supportive of your client's goals.
# Keep responses upbeat and actionable."""

# VAD settings
SILENCE_DURATION = 4.0  # Seconds of silence to stop recording
MIN_SPEECH_DURATION = 0.8  # Minimum speech length
MAX_RECORDING_DURATION = 30  # Maximum recording length

# Audio settings
SAMPLE_RATE = 16000
CHUNK = 512

MAX_HISTORY = 10

# ============================================
# INITIALIZE
# ============================================

print("="*60)
print("🎙️  INITIALIZING ALWAYS-LISTENING CHATBOT")
print("="*60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ Voice sample not found: {VOICE_SAMPLE}")
    exit(1)

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")

# Load Silero VAD
print("\n⏳ Loading Voice Activity Detection...")
start = time.time()
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, read_audio, *_) = utils
print(f"✅ VAD loaded in {time.time() - start:.1f}s")

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

# ============================================
# FUNCTIONS
# ============================================

def listen_continuously(audio_queue, stop_event):
    """Background thread: continuously listen for speech"""
    
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
    
    print("\n👂 Always listening... Speak naturally!")
    print("   (Say 'goodbye' or 'stop listening' to exit)")
    
    frames = []
    speech_started = False
    silence_chunks = 0
    silence_threshold = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)
    min_speech_chunks = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK)
    max_chunks = int(MAX_RECORDING_DURATION * SAMPLE_RATE / CHUNK)
    
    chunks_recorded = 0
    
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert to tensor for VAD
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            
            # Check if speech is present
            speech_prob = model_vad(audio_tensor, SAMPLE_RATE).item()
            
            # Speech threshold
            if speech_prob > 0.5:
                if not speech_started:
                    speech_started = True
                    print("\n🗣️  Speech detected! Recording...")
                    frames = []
                    chunks_recorded = 0
                
                frames.append(data)
                chunks_recorded += 1
                silence_chunks = 0
                
                # Safety: stop if too long
                if chunks_recorded >= max_chunks:
                    speech_started = False
                    print("⏸️  Max duration reached")
                    # Process this recording
                    if chunks_recorded >= min_speech_chunks:
                        audio_queue.put(frames[:])
                    frames = []
            else:
                if speech_started:
                    frames.append(data)
                    chunks_recorded += 1
                    silence_chunks += 1
                    
                    # Stop if enough silence
                    if silence_chunks > silence_threshold:
                        if chunks_recorded >= min_speech_chunks:
                            print("✅ Speech ended")
                            # Send recording for processing
                            audio_queue.put(frames[:])
                        
                        speech_started = False
                        frames = []
                        silence_chunks = 0
                        chunks_recorded = 0
        
        except Exception as e:
            print(f"⚠️  Audio error: {e}")
            continue
    
    stream.stop_stream()
    stream.close()
    p.terminate()

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
    
    # Build the prompt with system instruction and conversation history
    messages = [SYSTEM_PROMPT]  # Start with system prompt
    
    # Add conversation history (last MAX_HISTORY exchanges)
    for msg in conversation_history[-MAX_HISTORY:]:
        messages.append(msg)
    
    # Add current user input
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

def run_always_listening_chatbot():
    """Always-listening chatbot"""
    
    print("\n" + "="*60)
    print("🎙️  ALWAYS-LISTENING CHATBOT - Ready!")
    print("="*60)
    print(f"Voice Model: {VOICE_SAMPLE}")
    print(f"LLM: {OLLAMA_MODEL}")
    print(f"VAD: Auto-detects speech, stops after {SILENCE_DURATION}s silence")
    print(f"\n🎭 Personality: {SYSTEM_PROMPT[:80]}...")
    print("\nJust start speaking naturally!")
    print("Say 'goodbye', 'stop listening', or press Ctrl+C to exit")
    print("="*60)
    
    conversation_history = []
    turn = 0
    
    # Queue for audio from listener thread
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start listening thread
    listener_thread = threading.Thread(
        target=listen_continuously,
        args=(audio_queue, stop_event),
        daemon=True
    )
    listener_thread.start()
    
    try:
        while True:
            # Wait for audio from listener
            frames = audio_queue.get()
            
            turn += 1
            start_total = time.time()
            
            # Save audio
            audio_file = os.path.join(OUTPUT_DIR, f"recording_{int(time.time())}.wav")
            save_audio(frames, audio_file)
            
            # Transcribe
            user_input = transcribe_audio(audio_file)
            print(f"\n💬 You said: \"{user_input}\"")
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'stop listening', 'exit', 'quit']):
                print("\n👋 Goodbye!")
                stop_event.set()
                break
            
            if not user_input:
                print("⚠️  No speech detected")
                turn -= 1
                continue
            
            # Get LLM response (now with system prompt)
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
                
                print("\n🔊 Playing response...")
                play_audio(audio_path)
            
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response_text}")
            
            print("\n" + "-"*60)
            print("👂 Listening...")
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Goodbye! Had {turn} conversations.")
        stop_event.set()

if __name__ == "__main__":
    run_always_listening_chatbot()