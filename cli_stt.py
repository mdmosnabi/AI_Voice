
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

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
        "FFmpeg was not found. Install imageio-ffmpeg "
        "or add FFmpeg to PATH."
    )


# --------------------------------------------------
# Load the new local model once
# --------------------------------------------------

def get_model():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Moonshine model directory not found:\n{MODEL_DIR}"
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

    _model.generation_config.decoder_start_token_id = (
        _tokenizer.cls_token_id
    )
    _model.generation_config.bos_token_id = (
        _tokenizer.cls_token_id
    )
    _model.generation_config.eos_token_id = (
        _tokenizer.sep_token_id
    )
    _model.generation_config.pad_token_id = (
        _tokenizer.pad_token_id
    )

    return _model, _tokenizer


# --------------------------------------------------
# Convert audio/video to temporary WAV
# --------------------------------------------------

def convert_to_wav(input_path: Path, wav_path: Path) -> None:
    command = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(input_path),
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
            result.stderr.strip() or "Audio conversion failed."
        )

    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        raise RuntimeError("FFmpeg did not produce a valid WAV file.")


# --------------------------------------------------
# Transcribe one 30-second audio chunk
# --------------------------------------------------

def transcribe_chunk(audio: np.ndarray) -> str:
    model, tokenizer = get_model()

    # Pad the waveform to a multiple of 320 samples.
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
# Public transcription function
# --------------------------------------------------

def transcribe_file(file_path: str | Path) -> str:
    source = Path(file_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Input file does not exist:\n{source}"
        )

    temp = tempfile.NamedTemporaryFile(
        prefix="cli_stt_",
        suffix=".wav",
        delete=False,
    )

    wav_path = Path(temp.name)
    temp.close()

    try:
        convert_to_wav(source, wav_path)

        transcripts = []
        chunk_frames = SAMPLE_RATE * CHUNK_SECONDS

        # Read the WAV incrementally, keeping memory usage lower.
        with wave.open(str(wav_path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getframerate() != SAMPLE_RATE
                or wav.getsampwidth() != 2
            ):
                raise RuntimeError(
                    "Expected 16 kHz, mono, 16-bit PCM WAV."
                )

            while True:
                raw_audio = wav.readframes(chunk_frames)

                if not raw_audio:
                    break

                audio_int16 = np.frombuffer(
                    raw_audio,
                    dtype="<i2",
                )

                audio_float = (
                    audio_int16.astype(np.float32) / 32768.0
                )

                text = transcribe_chunk(audio_float)

                if text:
                    transcripts.append(text)

        return " ".join(transcripts).strip()

    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local Bengali speech-to-text for audio and video."
    )

    parser.add_argument(
        "file",
        help="Path to an audio or video file.",
    )

    args = parser.parse_args()

    try:
        transcript = transcribe_file(args.file)

        # Print only the final, combined transcript.
        print(transcript)

        return 0

    except Exception as exc:
        print(f"[STT ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
