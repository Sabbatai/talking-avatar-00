# chatbot_voice.py - Ollama + XTTS Voice Cloning Chatbot

import requests
import time
import os
from TTS.api import TTS

# ============================================
# CONFIGURATION
# ============================================

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Voice cloning settings
VOICE_SAMPLE = "voices/hoan_talking_0.wav"  # ← UPDATE THIS to your actual file
OUTPUT_DIR = "outputs"

# Conversation settings
MAX_HISTORY = 10

# ============================================
# INITIALIZE
# ============================================

print("="*60)
print("🚀 INITIALIZING VOICE CHATBOT")
print("="*60)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check voice sample exists
if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ ERROR: Voice sample not found at {VOICE_SAMPLE}")
    print("\nPlease place your voice sample in the 'voices' folder")
    print("and update VOICE_SAMPLE in this script.")
    exit(1)

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")

# Load TTS model (only once at startup)
print("\n⏳ Loading XTTS v2 model (this takes ~10-20 seconds)...")
start = time.time()

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print(f"✅ Model loaded in {time.time() - start:.1f}s")
except Exception as e:
    print(f"❌ Failed to load TTS model: {e}")
    print("\nTrying to load on CPU instead (slower)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print(f"✅ Model loaded on CPU in {time.time() - start:.1f}s")

# ============================================
# FUNCTIONS
# ============================================

def test_ollama_connection():
    """Test if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def get_llm_response(user_input, conversation_history):
    """Get response from Ollama LLM"""
    
    # Build conversation context
    context = "\n".join(conversation_history[-MAX_HISTORY:])
    prompt = f"{context}\nUser: {user_input}\nAssistant:"
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Error: LLM returned status {response.status_code}"
            
    except Exception as e:
        return f"Error calling LLM: {e}"

def generate_voice(text, output_filename):
    """Generate voice using XTTS v2"""
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"🎙️  Generating voice...", end='', flush=True)
    start = time.time()
    
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=VOICE_SAMPLE,
            file_path=output_path,
            language="en"
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
        # Try to play with Windows Media Player
        subprocess.Popen(["wmplayer", audio_path], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
    except:
        # If that fails, just tell user where file is
        print(f"   💡 Audio saved: {audio_path}")

# ============================================
# MAIN CHATBOT LOOP
# ============================================

def run_chatbot():
    """Main interactive chatbot"""
    
    # Test Ollama connection
    print("\n🔍 Testing Ollama connection...")
    if not test_ollama_connection():
        print("❌ Ollama is not running!")
        print("\nPlease start Ollama:")
        print("  1. Open a new terminal")
        print("  2. Run: ollama serve")
        print("  3. Then run this script again")
        return
    
    print("✅ Ollama is running")
    
    print("\n" + "="*60)
    print("🎙️  VOICE CHATBOT - Ready!")
    print("="*60)
    print(f"Voice Model: {VOICE_SAMPLE}")
    print(f"LLM: {OLLAMA_MODEL}")
    print(f"Output: {OUTPUT_DIR}/")
    print("\nType your message (or 'quit' to exit)")
    print("="*60 + "\n")
    
    conversation_history = []
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye!")
                print(f"\nGenerated {conversation_count} responses")
                print(f"Audio files saved in: {OUTPUT_DIR}/")
                break
            
            if not user_input:
                continue
            
            conversation_count += 1
            
            # Step 1: Get LLM response
            print("🤔 Thinking...", end='', flush=True)
            start_total = time.time()
            start = time.time()
            
            response_text = get_llm_response(user_input, conversation_history)
            llm_time = time.time() - start
            print(f"\r✅ Response in {llm_time:.1f}s")
            
            # Show response text
            print(f"\n🤖 Avatar: {response_text}\n")
            
            # Step 2: Generate voice
            audio_filename = f"response_{conversation_count:03d}_{int(time.time())}.wav"
            audio_path = generate_voice(response_text, audio_filename)
            
            if audio_path:
                total_time = time.time() - start_total
                print(f"⏱️  Total time: {total_time:.1f}s (LLM: {llm_time:.1f}s + TTS: {total_time-llm_time:.1f}s)")
                
                # Auto-play audio
                play_audio(audio_path)
            
            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response_text}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            print(f"\nGenerated {conversation_count} responses")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ============================================
# START
# ============================================

if __name__ == "__main__":
    run_chatbot()