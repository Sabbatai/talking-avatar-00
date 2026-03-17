# chatbot_f5.py - Ollama + F5-TTS Interactive Avatar Chatbot

import requests
import json
import time
import os
from f5_tts.api import F5TTS

# ============================================
# CONFIGURATION
# ============================================

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Voice settings
VOICE_SAMPLE = "voices/hoan_talking_0.wav"  # ← UPDATE if different
OUTPUT_DIR = "outputs"

# Conversation settings
MAX_HISTORY = 10

# ============================================
# INITIALIZE
# ============================================

print("="*60)
print("🚀 INITIALIZING F5-TTS CHATBOT")
print("="*60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check voice sample
if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ Voice sample not found: {VOICE_SAMPLE}")
    exit(1)

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")

# Load F5-TTS model
print("\n⏳ Loading F5-TTS model...")
start = time.time()
tts = F5TTS()
print(f"✅ F5-TTS loaded in {time.time() - start:.1f}s")

# ============================================
# FUNCTIONS
# ============================================

def test_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_llm_response(user_input, history):
    """Get response from Ollama with streaming"""
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
        
        print()  # New line
        return full_response.strip()
        
    except Exception as e:
        print(f"\n❌ LLM Error: {e}")
        return "Sorry, I encountered an error."

def generate_voice(text, filename):
    """Generate voice using F5-TTS"""
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    print(f"\n🎙️  Generating voice...", end='', flush=True)
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

# ============================================
# MAIN CHATBOT
# ============================================

def run_chatbot():
    """Main interactive loop"""
    
    # Test Ollama
    print("\n🔍 Testing Ollama connection...")
    if not test_ollama():
        print("❌ Ollama not running!")
        print("\nStart Ollama in another terminal:")
        print("  ollama serve")
        return
    
    print("✅ Ollama connected")
    
    print("\n" + "="*60)
    print("🎙️  F5-TTS CHATBOT - Ready!")
    print("="*60)
    print(f"Voice: {VOICE_SAMPLE}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Output: {OUTPUT_DIR}/")
    print("\nTips:")
    print("  - Type naturally")
    print("  - Use <Laughter:1> or <Laughter:2> for laughs")
    print("  - Type 'quit' to exit")
    print("="*60)
    
    conversation_history = []
    count = 0
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"\n👋 Goodbye! Generated {count} responses.")
                break
            
            if not user_input:
                continue
            
            count += 1
            start_total = time.time()
            
            # Step 1: Get LLM response (streaming)
            print("🤔 Thinking...", end='', flush=True)
            start_llm = time.time()
            
            response_text = get_llm_response(user_input, conversation_history)
            llm_time = time.time() - start_llm
            
            # Step 2: Generate voice
            audio_filename = f"response_{count:03d}_{int(time.time())}.wav"
            audio_path = generate_voice(response_text, audio_filename)
            
            if audio_path:
                total_time = time.time() - start_total
                tts_time = total_time - llm_time
                
                print(f"⏱️  Total: {total_time:.1f}s (LLM: {llm_time:.1f}s + TTS: {tts_time:.1f}s)")
                print(f"📁 {audio_path}")
                
                # Try to auto-play
                try:
                    import subprocess
                    subprocess.Popen(
                        ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except:
                    pass
            
            # Update history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response_text}")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye! Generated {count} responses.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    run_chatbot()