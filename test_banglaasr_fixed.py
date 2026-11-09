
from pathlib import Path
import time

import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(
    r"D:\AI_Voice\models\banglaasr"
    r"\models--bangla-speech-processing--BanglaASR"
    r"\snapshots\89c3a1b9585be48d9527cd8964c150fe2eebdbaa"
)

AUDIO_PATH = Path(r"D:\AI_Voice\outputs\real_test.wav")

SAMPLE_RATE = 16000

# First chunk: 10 seconds
FIRST_CHUNK_SECONDS = 10

# Remaining chunks: 12 seconds
CHUNK_SECONDS = 12

# Overlap between chunks: 2 seconds
OVERLAP_SECONDS = 2

MAX_NEW_TOKENS = 256
CPU_THREADS = 4


# ============================================================
# CPU CONFIGURATION
# ============================================================

torch.set_num_threads(CPU_THREADS)


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Model directory not found:\n{MODEL_DIR}"
    )

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("Loading BanglaASR model...")
print("=" * 65)

load_start = time.perf_counter()

processor = WhisperProcessor.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

model = WhisperForConditionalGeneration.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

model.eval()
model.to("cpu")

print(
    f"Model loaded in "
    f"{time.perf_counter() - load_start:.2f} seconds"
)


# ============================================================
# LOAD AUDIO
# ============================================================

print("\nLoading audio...")

audio, _ = librosa.load(
    str(AUDIO_PATH),
    sr=SAMPLE_RATE,
    mono=True,
)

audio = np.asarray(audio, dtype=np.float32)

if audio.size == 0:
    raise ValueError("Audio file is empty.")

total_samples = len(audio)
total_duration = total_samples / SAMPLE_RATE


# ============================================================
# CREATE CHUNKS
# ============================================================

first_chunk_size = FIRST_CHUNK_SECONDS * SAMPLE_RATE
chunk_size = CHUNK_SECONDS * SAMPLE_RATE

# Subsequent chunks start 10 seconds apart:
# 12-second chunk minus 2-second overlap.
step_size = (CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE

chunks = []

# First chunk: 0-10 seconds
first_end = min(first_chunk_size, total_samples)

chunks.append((0, first_end))

# Remaining chunks:
# 8-20, 18-30, 28-40, ...
if total_samples > first_end:
    start = (FIRST_CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE

    while start < total_samples:
        end = min(start + chunk_size, total_samples)

        chunks.append((start, end))

        start += step_size


print("\n" + "=" * 65)
print("AUDIO INFORMATION")
print("=" * 65)

print(f"Audio file       : {AUDIO_PATH.name}")
print(f"Audio duration   : {total_duration:.2f} seconds")
print(f"Sample rate      : {SAMPLE_RATE} Hz")
print(f"Total chunks     : {len(chunks)}")
print(f"First chunk      : {FIRST_CHUNK_SECONDS} seconds")
print(f"Other chunks     : {CHUNK_SECONDS} seconds")
print(f"Overlap          : {OVERLAP_SECONDS} seconds")
print(f"Device           : CPU")
print("=" * 65)


# ============================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================

all_transcripts = []
chunk_times = []

total_start = time.perf_counter()

for i, (start, end) in enumerate(chunks, start=1):

    chunk = audio[start:end]

    chunk_start_sec = start / SAMPLE_RATE
    chunk_end_sec = end / SAMPLE_RATE
    chunk_duration = len(chunk) / SAMPLE_RATE

    print(
        f"\n\n{'-' * 65}\n"
        f"Processing chunk {i}/{len(chunks)}\n"
        f"Audio range: {chunk_start_sec:.1f}s - "
        f"{chunk_end_sec:.1f}s\n"
        f"Chunk duration: {chunk_duration:.1f}s\n"
        f"{'-' * 65}"
    )

    # Convert audio to Whisper input features
    inputs = processor(
        chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to("cpu")

    inference_start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            input_features,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
        )

    elapsed = time.perf_counter() - inference_start

    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()

    all_transcripts.append(text)
    chunk_times.append(elapsed)

    print("\nTranscript:")
    print(text if text else "[No text detected]")

    print(f"\nProcessing time: {elapsed:.2f} seconds")


# ============================================================
# PRINT INDIVIDUAL CHUNK RESULTS
# ============================================================

print("\n\n" + "=" * 65)
print("INDIVIDUAL CHUNK TRANSCRIPTS")
print("=" * 65)

for i, ((start, end), text) in enumerate(
    zip(chunks, all_transcripts),
    start=1,
):
    print(f"\n[Chunk {i}]")
    print(
        f"Time: {start / SAMPLE_RATE:.1f}s - "
        f"{end / SAMPLE_RATE:.1f}s"
    )
    print(text if text else "[No text detected]")


# ============================================================
# COMBINED TRANSCRIPT
# ============================================================

print("\n\n" + "=" * 65)
print("COMBINED TRANSCRIPT")
print("=" * 65)

# Note:
# This joins the chunk outputs as-is.
# Since chunks overlap, some words may appear more than once.
full_transcript = " ".join(
    text for text in all_transcripts if text
)

print(full_transcript if full_transcript else "[No text detected]")


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

total_processing_time = time.perf_counter() - total_start

print("\n\n" + "=" * 65)
print("PERFORMANCE SUMMARY")
print("=" * 65)

print(f"Audio duration        : {total_duration:.2f} seconds")
print(f"Chunks processed      : {len(all_transcripts)}/{len(chunks)}")
print(f"Total inference time  : {sum(chunk_times):.2f} seconds")
print(f"Total elapsed time    : {total_processing_time:.2f} seconds")

if chunk_times:
    print(f"Average chunk time    : {np.mean(chunk_times):.2f} seconds")

print("=" * 65)
print("Transcription finished.")
