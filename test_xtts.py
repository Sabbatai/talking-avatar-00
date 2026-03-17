# test_xtts.py - Test Coqui XTTS v2 voice cloning

from TTS.api import TTS
import time
import os

print("="*60)
print("Testing Coqui XTTS v2 Voice Cloning")
print("="*60)

# Path to your voice sample
VOICE_SAMPLE = "voices/hoan_talking_0.wav"  # ← UPDATE THIS with your actual filename

# Check if voice sample exists
if not os.path.exists(VOICE_SAMPLE):
    print(f"\n❌ ERROR: Voice sample not found at {VOICE_SAMPLE}")
    print("\nPlease:")
    print("1. Place your voice sample in the 'voices' folder")
    print("2. Update VOICE_SAMPLE in this script to match the filename")
    exit(1)

# Test text
TEST_TEXT = "Hello! This is a test of the voice cloning system. How does it sound?"

print(f"\n✅ Voice sample found: {VOICE_SAMPLE}")
print(f"\n🔄 Loading XTTS v2 model...")
print("   (First run: downloads ~2GB, takes 30-60 seconds)")
print("   (Subsequent runs: loads from cache, ~10-20 seconds)")

start = time.time()

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

load_time = time.time() - start
print(f"\n✅ Model loaded in {load_time:.1f} seconds")

print(f"\n🎙️  Cloning voice from: {VOICE_SAMPLE}")
print(f"📝 Text: '{TEST_TEXT}'")
print("\n⏳ Generating speech...", end='', flush=True)

start = time.time()

# Generate speech
tts.tts_to_file(
    text=TEST_TEXT,
    speaker_wav=VOICE_SAMPLE,
    file_path="outputs/test_output.wav",
    language="en"
)

gen_time = time.time() - start
print(f"\r✅ Speech generated in {gen_time:.1f} seconds")

print(f"\n🔊 Audio saved to: outputs/test_output.wav")
print("\n✅ Test complete! Listen to the audio file to verify voice cloning.")
print("="*60)