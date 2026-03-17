# test_f5tts_official.py - Test F5-TTS

import torch
from f5_tts.api import F5TTS
import time
import os

print("="*60)
print("F5-TTS Installation Test")
print("="*60)

# Check setup
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

# Configuration
VOICE_SAMPLE = "voices/hoan_talking_0.wav"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ Voice sample not found: {VOICE_SAMPLE}")
    print("Update VOICE_SAMPLE path or add a voice sample to voices/")
    exit(1)

print(f"\n✅ Voice sample: {VOICE_SAMPLE}")

# Test cases
tests = [
    ("Basic", "Hello! This is a test of F5-TTS with natural prosody."),
    ("Laughter", "That's hilarious! <Laughter:2> I can't stop laughing!"),
    ("Emotion", "I'm so excited! <Laughter:1> This is really working!")
]

try:
    print("\n⏳ Loading F5-TTS model (first run downloads ~1-2GB)...")
    start = time.time()
    
    tts = F5TTS()
    
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    # Run tests
    for i, (name, text) in enumerate(tests, 1):
        print(f"\n--- Test {i}: {name} ---")
        print(f"Text: {text}")
        print("🎙️  Generating...", end='', flush=True)
        
        start = time.time()
        output_file = os.path.join(OUTPUT_DIR, f"f5_test_{i}.wav")
        
        tts.infer(
            ref_file=VOICE_SAMPLE,
            ref_text="",  # Auto-transcribe
            gen_text=text,
            file_wave=output_file,
        )
        
        gen_time = time.time() - start
        print(f"\r✅ Generated in {gen_time:.1f}s")
        print(f"📁 {output_file}")
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("\nListen to the files in outputs/ to check:")
    print("  - Voice cloning quality")
    print("  - Natural prosody")
    print("  - Laughter expression")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nCommon issues:")
    print("1. FFmpeg not installed: choco install ffmpeg")
    print("2. CUDA not available: check nvidia-smi")
    print("3. Voice sample missing or invalid format")
    import traceback
    traceback.print_exc()