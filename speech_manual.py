from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_DIR = Path(r"D:\AI_Voice\models\moonshine-base-bn")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
MAX_NEW_TOKENS = 200
CPU_THREADS = 4

_model = None
_tokenizer = None

torch.set_num_threads(CPU_THREADS)


# --------------------------------------------------
# Windows file picker
# --------------------------------------------------

def choose_media_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected = filedialog.askopenfilename(
            title="Select an audio or video file",
            filetypes=[
                (
                    "Audio and video files",
                    "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus "
                    "*.wma *.mp4 *.mkv *.mov *.avi *.webm *.mpeg *.mpg",
                ),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return Path(selected) if selected else None


# --------------------------------------------------
# Locate FFmpeg
# --------------------------------------------------

def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(path).is_file():
            return path
    except (ImportError, RuntimeError, OSError):
        pass

    path = shutil.which("ffmpeg")
    if path:
        return path

    raise RuntimeError(
        "FFmpeg was not found. Install imageio-ffmpeg in the active "
        "Python environment or add FFmpeg to PATH."
    )


# --------------------------------------------------
# Load the local Moonshine model once
# --------------------------------------------------

def get_model():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print("[1/5] Loading Moonshine Bengali model...", flush=True)

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model directory not found:\n{MODEL_DIR}"
        )

    _tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        trust_remote_code=False,
    )

    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    _model.to("cpu")
    _model.eval()

    _model.generation_config.decoder_start_token_id = _tokenizer.cls_token_id
    _model.generation_config.bos_token_id = _tokenizer.cls_token_id
    _model.generation_config.eos_token_id = _tokenizer.sep_token_id
    _model.generation_config.pad_token_id = _tokenizer.pad_token_id

    print("[1/5] Model loaded successfully.", flush=True)
    return _model, _tokenizer


# --------------------------------------------------
# Convert selected media to temporary 16 kHz mono WAV
# --------------------------------------------------

def convert_to_wav(source: Path, wav_path: Path) -> None:
    print("[2/5] Converting selected file to 16 kHz mono WAV...", flush=True)

    command = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(source),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(wav_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg conversion failed:\n"
            + (result.stderr.strip() or "Unknown FFmpeg error.")
        )

    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        raise RuntimeError("FFmpeg did not produce a valid WAV file.")

    print("[2/5] Audio conversion completed.", flush=True)


# --------------------------------------------------
# Transcribe one chunk (maximum 30 seconds)
# --------------------------------------------------

def transcribe_chunk(audio: np.ndarray) -> str:
    model, tokenizer = get_model()

    # Moonshine model's current input handling: pad to a multiple of 320.
    padding = (-len(audio)) % 320
    if padding:
        audio = np.pad(audio, (0, padding))

    input_values = torch.from_numpy(
        np.asarray(audio, dtype=np.float32)
    ).unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_values=input_values,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            decoder_start_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()


# --------------------------------------------------
# Run manual transcription
# --------------------------------------------------

def run() -> int:
    print("=" * 64)
    print("        MANUAL BENGALI AUDIO / VIDEO TRANSCRIBER")
    print("=" * 64)
    print("Opening Windows file picker...", flush=True)

    source = choose_media_file()
    if source is None:
        print("No file selected. Exiting.")
        return 0

    print(f"Selected file: {source}", flush=True)

    if not source.is_file():
        raise FileNotFoundError(f"Selected file does not exist:\n{source}")

    # Load model first so model errors are reported before conversion.
    get_model()

    temp_file = tempfile.NamedTemporaryFile(
        prefix="manual_stt_",
        suffix=".wav",
        delete=False,
    )
    wav_path = Path(temp_file.name)
    temp_file.close()

    try:
        convert_to_wav(source, wav_path)

        print("[3/5] Reading WAV and preparing 30-second chunks...", flush=True)
        chunk_frames = SAMPLE_RATE * CHUNK_SECONDS
        transcripts: list[str] = []

        with wave.open(str(wav_path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getframerate() != SAMPLE_RATE
                or wav.getsampwidth() != 2
            ):
                raise RuntimeError(
                    "Converted WAV is not 16 kHz mono 16-bit PCM."
                )

            total_frames = wav.getnframes()
            total_seconds = total_frames / SAMPLE_RATE
            total_chunks = (total_frames + chunk_frames - 1) // chunk_frames

            print(f"Audio duration: {total_seconds:.1f} seconds", flush=True)
            print(f"Chunks to process: {total_chunks}", flush=True)
            print("[4/5] Transcribing chunks. Please wait...", flush=True)

            chunk_number = 0
            while True:
                raw_audio = wav.readframes(chunk_frames)
                if not raw_audio:
                    break

                chunk_number += 1
                audio_int16 = np.frombuffer(raw_audio, dtype="<i2")
                audio_float = audio_int16.astype(np.float32) / 32768.0
                duration = len(audio_float) / SAMPLE_RATE

                print(
                    f"  Processing chunk {chunk_number}/{total_chunks} "
                    f"({duration:.1f} seconds)...",
                    flush=True,
                )

                chunk_text = transcribe_chunk(audio_float)
                transcripts.append(chunk_text)

                print(
                    f"  Chunk {chunk_number}/{total_chunks} completed.",
                    flush=True,
                )

        final_text = " ".join(text for text in transcripts if text).strip()

        print("[5/5] All chunks completed. Combining transcript...", flush=True)
        print("\n" + "=" * 64)
        print("FINAL TRANSCRIPT")
        print("=" * 64)
        print(final_text if final_text else "[No speech recognized]")
        print("=" * 64)

        return 0

    finally:
        try:
            wav_path.unlink(missing_ok=True)
            print("Temporary WAV cleaned up.", flush=True)
        except OSError as exc:
            print(f"Warning: could not remove temporary WAV: {exc}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("\nCancelled by user.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", flush=True)
        raise SystemExit(1)
