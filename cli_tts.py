
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

# Existing Hugging Face models
MODEL_MAP = {
    "ben_male": "facebook/mms-tts-ben",
    "hin_male": "facebook/mms-tts-hin",
    "hin_female": "Anjan9320/fb-mms-tts-hin-ft-female",
    "eng_male": "facebook/mms-tts-eng",
}

# New local Bangladeshi Bangla female VITS model
LOCAL_VITS_PROFILES = {
    "ben_female": {
        "model_dir": Path(
            r"D:\AI_Voice\models\bangladeshi_bangla_tts_vits"
        ),
        "checkpoint": "pytorch_model.pth",
        "config": "config.json",
    },
}

ALL_VOICES = list(MODEL_MAP.keys()) + list(LOCAL_VITS_PROFILES.keys())

OUTPUT_DIR = BASE_DIR / "outputs"


# ==================================================
# UTF-8 CONSOLE SUPPORT
# ==================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ==================================================
# EXISTING TRANSFORMERS TTS ENGINE
# ==================================================

def generate_transformers_audio(
    text: str,
    model_id: str,
    out_path: Path,
):
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

    wavfile.write(
        str(out_path),
        rate=model.config.sampling_rate,
        data=audio_data,
    )


# ==================================================
# NEW LOCAL BANGLA FEMALE VITS ENGINE
# ==================================================

def generate_bengali_female_audio(
    text: str,
    out_path: Path,
):
    profile = LOCAL_VITS_PROFILES["ben_female"]

    model_dir = profile["model_dir"]
    model_path = model_dir / profile["checkpoint"]
    config_path = model_dir / profile["config"]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Bangla VITS model weights not found: {model_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Bangla VITS config not found: {config_path}"
        )

    print("[+] Loading local Bangladeshi Bangla Female VITS...")
    print("[+] Device: CPU")

    from TTS.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(
        tts_checkpoint=str(model_path),
        tts_config_path=str(config_path),
        use_cuda=False,
    )

    print("[+] Generating Bengali female voice...")

    audio = synthesizer.tts(text)

    synthesizer.save_wav(
        audio,
        str(out_path),
    )


# ==================================================
# SHARED CLI GENERATION FUNCTION
# ==================================================

def generate_audio(
    text: str,
    model_key: str,
    output_filename: str,
):
    if model_key not in ALL_VOICES:
        raise ValueError(
            f"Invalid voice '{model_key}'. "
            f"Available voices: {ALL_VOICES}"
        )

    # Prevent output filenames from escaping outputs/
    filename = Path(output_filename)

    if (
        not output_filename.strip()
        or filename.name != output_filename
        or filename.suffix.lower() != ".wav"
        or output_filename in {".", ".."}
    ):
        raise ValueError(
            "Output must be a WAV filename, e.g. output.wav"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (OUTPUT_DIR / filename.name).resolve()

    print(f"\n[+] Selected Profile: {model_key}")
    print(f"[+] Output: {out_path}")

    start_time = time.perf_counter()

    if model_key in LOCAL_VITS_PROFILES:
        generate_bengali_female_audio(
            text=text,
            out_path=out_path,
        )
    else:
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
    parser = argparse.ArgumentParser(
        description="Local TTS Engine CLI for project integration"
    )

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
        default="output.wav",
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
