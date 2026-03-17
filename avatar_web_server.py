"""
Web Server for AI Avatar Chat
Bridges browser audio input to chatbot pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import time
import uuid
import subprocess
from pathlib import Path
import requests

# Whisper and TTS imports
from faster_whisper import WhisperModel
from f5_tts.api import F5TTS

app = FastAPI()

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
WAV2LIP_SERVER_URL = "http://localhost:8000"
VOICE_SAMPLE = "voices/hoan_talking_0.wav"
AVATAR_PATH = "avatars/hoan_10s.mp4"
OUTPUT_DIR = "web_outputs"
TEMP_DIR = "temp_web"

# System prompt
SYSTEM_PROMPT = """You are an AI helping write dialogue for a fictional character in a tech thriller. 
The character is a charismatic but morally ambiguous tech CEO - confident, visionary, and somewhat ruthless in business.
Respond as this character would: ambitious, strategic, occasionally dismissive of regulations, focused on "disruption" and "the future."
Keep responses to 2-3 sentences. Make them sound like a TED talk meets corporate villain monologue."""

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("="*60)
print("🌐 INITIALIZING WEB SERVER FOR AI AVATAR")
print("="*60)

# Load models
print("\n⏳ Loading Whisper...")
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print("✅ Whisper loaded")

print("\n⏳ Loading F5-TTS...")
tts = F5TTS()
print("✅ F5-TTS loaded")

print("\n✅ Web server ready!")
print(f"📁 Outputs: {OUTPUT_DIR}")
print(f"🎤 Voice: {VOICE_SAMPLE}")
print(f"🎬 Avatar: {AVATAR_PATH}")

# Conversation history (in-memory, per-session in production)
conversation_history = []

def convert_webm_to_wav(webm_path: str, wav_path: str):
    """Convert WebM audio to WAV using FFmpeg"""
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', webm_path, '-ar', '16000', '-ac', '1', wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion error: {e}")
        return False

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio with Whisper"""
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def get_llm_response(user_input: str) -> str:
    """Get response from Ollama"""
    messages = [SYSTEM_PROMPT]
    
    for msg in conversation_history[-10:]:  # Last 10 messages
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
        print(f"LLM Error: {e}")
        return "I apologize, I'm having trouble formulating a response right now."

def generate_voice(text: str, output_path: str) -> bool:
    """Generate voice with F5-TTS"""
    try:
        tts.infer(
            ref_file=VOICE_SAMPLE,
            ref_text="",
            gen_text=text,
            file_wave=output_path,
        )
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def generate_video(audio_path: str, output_path: str) -> bool:
    """Generate video with Wav2Lip server"""
    try:
        abs_audio_path = os.path.abspath(audio_path)
        abs_avatar_path = os.path.abspath(AVATAR_PATH)
        abs_output_path = os.path.abspath(output_path)
        
        response = requests.post(
            f"{WAV2LIP_SERVER_URL}/generate",
            json={
                "audio_path": abs_audio_path,
                "avatar_path": abs_avatar_path,
                "output_path": abs_output_path,
                "fps": 15
            },
            timeout=120
        )
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Wav2Lip Error: {e}")
        return False

@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    """
    Process audio from browser and return avatar response
    """
    session_id = str(uuid.uuid4())[:8]
    
    try:
        # Save uploaded audio
        webm_path = os.path.join(TEMP_DIR, f"{session_id}_input.webm")
        wav_path = os.path.join(TEMP_DIR, f"{session_id}_input.wav")
        
        with open(webm_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Convert to WAV
        if not convert_webm_to_wav(webm_path, wav_path):
            raise HTTPException(status_code=500, detail="Audio conversion failed")
        
        # Transcribe
        print(f"\n[{session_id}] Transcribing...")
        user_text = transcribe_audio(wav_path)
        print(f"[{session_id}] User: {user_text}")
        
        if not user_text:
            raise HTTPException(status_code=400, detail="No speech detected")
        
        # Get LLM response
        print(f"[{session_id}] Generating response...")
        avatar_text = get_llm_response(user_text)
        print(f"[{session_id}] Avatar: {avatar_text}")
        
        # Generate voice
        print(f"[{session_id}] Generating voice...")
        audio_output = os.path.join(OUTPUT_DIR, f"{session_id}_voice.wav")
        if not generate_voice(avatar_text, audio_output):
            raise HTTPException(status_code=500, detail="Voice generation failed")
        
        # Generate video
        print(f"[{session_id}] Generating video...")
        video_output = os.path.join(OUTPUT_DIR, f"{session_id}_response.mp4")
        if not generate_video(audio_output, video_output):
            raise HTTPException(status_code=500, detail="Video generation failed")
        
        # Update conversation history
        conversation_history.append(f"User: {user_text}")
        conversation_history.append(f"Assistant: {avatar_text}")
        
        # Cleanup temp files
        try:
            os.remove(webm_path)
            os.remove(wav_path)
        except:
            pass
        
        print(f"[{session_id}] ✅ Complete")
        
        return {
            "user_text": user_text,
            "avatar_text": avatar_text,
            "video_url": f"/video/{session_id}_response.mp4",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve generated video files"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(file_path, media_type="video/mp4")

@app.get("/")
async def root():
    """Serve the web interface"""
    return FileResponse("avatar_web_interface.html")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper": "loaded",
        "tts": "loaded",
        "wav2lip_server": WAV2LIP_SERVER_URL
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🌐 Starting Web Server on http://localhost:8001")
    print("="*60)
    print("\nOpen http://localhost:8001 in your browser to chat!")
    uvicorn.run(app, host="0.0.0.0", port=8001)
