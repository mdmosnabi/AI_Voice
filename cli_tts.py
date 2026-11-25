import argparse
import sys
import time
from pathlib import Path

import scipy.io.wavfile as wavfile
import torch
from transformers import AutoTokenizer, VitsModel

# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
INDIC_MODEL_DIR = BASE_DIR / "models" / "indic-tiny-tts-bengali"

DEFAULT_SPEED = 1.05
CPU_THREADS = 4

MODEL_MAP = {
    "hin_male": "facebook/mms-tts-hin",
    "hin_female": "Anjan9320/fb-mms-tts-hin-ft-female",
    "eng_male": "facebook/mms-tts-eng",
}

INDIC_VOICE_PROFILES = {
    "ben_male": "male",
    "ben_female": "female",
}

ALL_VOICES = list(MODEL_MAP.keys()) + list(INDIC_VOICE_PROFILES.keys())


# ==================================================
# UTF-8 CONSOLE SUPPORT
# ==================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ==================================================
# INDIC TINY TTS ENGINE
# ==================================================

def generate_indic_bengali_audio(text: str, voice: str, out_path: Path):
    if not INDIC_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Indic Tiny TTS model not found: {INDIC_MODEL_DIR}")

    if str(INDIC_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(INDIC_MODEL_DIR))

    from indic_tiny_tts import TinyTTS

    print("[+] Loading Indic Tiny TTS Bengali...")
    print("[+] Device: CPU")
    print(f"[+] Threads: {CPU_THREADS}")
    print(f"[+] Speed: {DEFAULT_SPEED}x")

    tts = TinyTTS.from_pretrained(
        str(INDIC_MODEL_DIR),
        device="cpu",
        threads=CPU_THREADS,
    )

    print(f"[+] Generating Bengali {voice} voice...")
    wav = tts.synthesize(text, voice=voice, speed=DEFAULT_SPEED)
    tts.save_wav(wav, str(out_path))


# ==================================================
# EXISTING TRANSFORMERS TTS ENGINE
# ==================================================

def generate_transformers_audio(text: str, model_id: str, out_path: Path):
    print(f"[+] Loading Transformers model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()

    torch.manual_seed(42)

    print("[+] Processing text...")
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio_data = output.squeeze().float().cpu().numpy()
    wavfile.write(str(out_path), rate=model.config.sampling_rate, data=audio_data)


# ==================================================
# SHARED CLI GENERATION FUNCTION
# ==================================================

def generate_audio(text: str, model_key: str, output_filename: str):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if model_key not in ALL_VOICES:
        raise ValueError(f"Invalid voice '{model_key}'. Available voices: {ALL_VOICES}")

    filename = Path(output_filename)

    if (
        not output_filename.strip()
        or filename.name != output_filename
        or filename.suffix.lower() != ".wav"
        or output_filename in {".", ".."}
    ):
        raise ValueError("Output must be a WAV filename, e.g. output.wav")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (OUTPUT_DIR / filename.name).resolve()

    print(f"\n[+] Selected Profile: {model_key}")
    print(f"[+] Output: {out_path}")

    start_time = time.perf_counter()

    if model_key in INDIC_VOICE_PROFILES:
        generate_indic_bengali_audio(
            text=text,
            voice=INDIC_VOICE_PROFILES[model_key],
            out_path=out_path,
        )
    elif model_key in MODEL_MAP:
        generate_transformers_audio(
            text=text,
            model_id=MODEL_MAP[model_key],
            out_path=out_path,
        )

    elapsed = time.perf_counter() - start_time

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError("Audio generation failed: output WAV is empty.")

    print(f"[+] Generation completed in {elapsed:.2f} seconds.")
    print(f"[+] File size: {out_path.stat().st_size / 1024:.1f} KB")

    # IMPORTANT: Keep absolute path as the final stdout line for Django.
    print(out_path, flush=True)

    return str(out_path)


# ==================================================
# COMMAND-LINE INTERFACE
# ==================================================

def main():
    parser = argparse.ArgumentParser(description="Local TTS Engine CLI")
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Text to convert into speech",
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        required=True,
        choices=ALL_VOICES,
        help=f"Voice profile: {', '.join(ALL_VOICES)}",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output WAV filename",
    )

    args = parser.parse_args()

    try:
        generate_audio(
            text=args.text,
            model_key=args.voice,
            output_filename=args.output,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


    