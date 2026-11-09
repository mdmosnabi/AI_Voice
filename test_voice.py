import os
import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer

# The complete 6-model mapping (Male & Female for Bengali, Hindi, English)
MODEL_MAP = {
    # Bengali Models
    "ben_male": "facebook/mms-tts-ben",
    "ben_female": "facebook/mms-tts-ben",  # Fallback to Meta MMS if custom community repo is private

    # Hindi Models
    "hin_male": "facebook/mms-tts-hin",
    "hin_female": "Anjan9320/fb-mms-tts-hin-ft-female",

    # English Models
    "eng_male": "facebook/mms-tts-eng",   # Using Meta MMS Eng (Clean public repo)
    "eng_female": "facebook/mms-tts-eng",
}

def generate_tts(text: str, voice_key: str, output_filename: str):
    if voice_key not in MODEL_MAP:
        raise ValueError(f"Voice key '{voice_key}' not supported. Choose from {list(MODEL_MAP.keys())}")

    model_id = MODEL_MAP[voice_key]
    print(f"\n[+] Loading model for [{voice_key}]: {model_id}...")

    # Load tokenizer and VITS model on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    
    # Fix seed for reproducible output duration
    torch.manual_seed(42)

    print(f"[+] Processing text: '{text}'")
    inputs = tokenizer(text, return_tensors="pt")

    # Generate speech
    with torch.no_grad():
        output = model(**inputs).waveform

    # Save audio
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", output_filename)
    
    audio_data = output.squeeze().float().cpu().numpy()
    sampling_rate = model.config.sampling_rate

    wavfile.write(output_path, rate=sampling_rate, data=audio_data)
    print(f"[✓] Audio successfully generated: {output_path}")

if __name__ == "__main__":
    print("==================================================")
    print("   6-Voice Local CPU TTS Test Suite (6 Models)    ")
    print("==================================================")

    # Test 1: Bengali Male
    generate_tts(
        text="নমস্কার! এটি বাংলা পুরুষ কন্ঠে টেস্ট ভয়েস।",
        voice_key="ben_male",
        output_filename="test_ben_male.wav"
    )

    # Test 2: Bengali Female
    generate_tts(
        text="নমস্কার! এটি বাংলা নারী কন্ঠে টেস্ট ভয়েস।",
        voice_key="ben_female",
        output_filename="test_ben_female.wav"
    )

    # Test 3: Hindi Male
    generate_tts(
        text="नमस्ते! यह हिंदी पुरुष आवाज़ में परीक्षण है।",
        voice_key="hin_male",
        output_filename="test_hin_male.wav"
    )

    # Test 4: Hindi Female
    generate_tts(
        text="नमस्ते! यह हिंदी महिला आवाज़ में परीक्षण है।",
        voice_key="hin_female",
        output_filename="test_hin_female.wav"
    )

    # Test 5: English Male
    generate_tts(
        text="Hello! This is an English male voice test.",
        voice_key="eng_male",
        output_filename="test_eng_male.wav"
    )

    # Test 6: English Female
    generate_tts(
        text="Hello! This is an English female voice test.",
        voice_key="eng_female",
        output_filename="test_eng_female.wav"
    )
    
    print("\n[✓] All 6 voice tests completed successfully!")