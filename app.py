import sys
import subprocess
from pathlib import Path

import imageio_ffmpeg
from pywhispercpp.model import Model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

INPUT_AUDIO = OUTPUT_DIR / "real_test.wav"
CONVERTED_AUDIO = OUTPUT_DIR / "real_test_whisper.wav"

MODEL_PATH = MODELS_DIR / "ggml-bengaliai-whisper-medium-q5_0.bin"

N_THREADS = 4


def exit_with_error(message):
    print(f"\n[Error] {message}")
    sys.exit(1)


def convert_audio():
    """Convert the input audio to mono, 16 kHz PCM WAV."""

    if not INPUT_AUDIO.is_file():
        exit_with_error(f"Audio file not found:\n{INPUT_AUDIO}")

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-i", str(INPUT_AUDIO),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(CONVERTED_AUDIO),
        ]

        print("\n[1/3] Converting audio to PCM WAV...")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            exit_with_error(
                "Audio conversion failed.\n"
                + result.stderr[-2500:]
            )

        if not CONVERTED_AUDIO.is_file():
            exit_with_error("Converted audio file was not created.")

        print("[OK] Audio converted successfully.")

    except FileNotFoundError as e:
        exit_with_error(f"FFmpeg could not be found: {e}")

    except Exception as e:
        exit_with_error(f"Audio conversion failed: {e}")


def load_model():
    """Load the Bengali-trained Whisper model."""

    print("\n[2/3] Loading Bengali-trained Whisper Medium...")
    print(f"Model: {MODEL_PATH}")

    if not MODEL_PATH.is_file():
        exit_with_error(
            "Model file not found.\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Check the model filename and models folder."
        )

    try:
        model = Model(
            str(MODEL_PATH),
            n_threads=N_THREADS,
            language="bn",
            translate=False,
        )

        print("[OK] Model loaded successfully.")
        return model

    except Exception as e:
        exit_with_error(f"Model loading failed: {e}")


def transcribe_audio(model):
    """Transcribe the audio and print the result in the console."""

    print("\n[3/3] Transcribing audio...")
    print("Target language: Bengali")
    print("Translation: Disabled")
    print("-" * 60)

    try:
        segments = model.transcribe(
            str(CONVERTED_AUDIO),
            language="bn",
            translate=False,
            no_context=True,
            temperature=0.0,
            temperature_inc=0.2,
            print_progress=True,
        )

        transcript_parts = []

        for segment in segments:
            text = getattr(segment, "text", "").strip()

            if text:
                transcript_parts.append(text)

        transcript = " ".join(transcript_parts).strip()

        print("\n")
        print("=" * 60)
        print("FULL TRANSCRIPTION")
        print("=" * 60)

        if transcript:
            print(transcript)
            print("\n[OK] Transcription completed.")
        else:
            print("[No speech detected]")
            print(
                "\nTry checking the audio quality or testing "
                "a different model."
            )

        print("=" * 60)

        # Intentionally do not save a transcript text file.
        return transcript

    except Exception as e:
        exit_with_error(f"Transcription failed: {e}")


def main():
    print("=" * 60)
    print("       LOCAL SPEECH-TO-TEXT")
    print("       Whisper.cpp + Bengali Model")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    convert_audio()
    model = load_model()
    transcribe_audio(model)


if __name__ == "__main__":
    main()