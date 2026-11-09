
from pathlib import Path
import time

import torch
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------
MODEL_DIR = Path(r"D:\AI_Voice\models\moonshine-base-bn")
AUDIO_PATH = Path(r"D:\AI_Voice\outputs\real_test.wav")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
MAX_NEW_TOKENS = 200

CPU_THREADS = 4

torch.set_num_threads(CPU_THREADS)


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------
if not MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

if not AUDIO_PATH.is_file():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")


# --------------------------------------------------
# LOAD TOKENIZER AND MODEL
# --------------------------------------------------
print("Loading Moonshine Bengali model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    trust_remote_code=False,
)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
).eval()

model.to("cpu")

# Configure decoder tokens
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
model.generation_config.bos_token_id = tokenizer.cls_token_id
model.generation_config.eos_token_id = tokenizer.sep_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id


# --------------------------------------------------
# LOAD ENTIRE AUDIO
# --------------------------------------------------
audio, _ = librosa.load(
    str(AUDIO_PATH),
    sr=SAMPLE_RATE,
    mono=True,
)

audio = np.asarray(audio, dtype=np.float32)

if audio.size == 0:
    raise ValueError("The audio file contains no samples.")

total_samples = len(audio)
total_duration = total_samples / SAMPLE_RATE

chunk_size = CHUNK_SECONDS * SAMPLE_RATE

total_chunks = (
    total_samples + chunk_size - 1
) // chunk_size

all_transcripts = []


# --------------------------------------------------
# TRANSCRIBE AUDIO IN 30-SECOND SECTIONS
# --------------------------------------------------
for index, start in enumerate(
    range(0, total_samples, chunk_size),
    start=1,
):
    end = min(start + chunk_size, total_samples)

    chunk = audio[start:end]
    chunk_duration = len(chunk) / SAMPLE_RATE

    # Pad to a multiple of 320 samples
    padding = (-len(chunk)) % 320

    if padding:
        chunk = np.pad(chunk, (0, padding))

    input_values = torch.from_numpy(
        np.asarray(chunk, dtype=np.float32)
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

    transcript = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    all_transcripts.append(transcript)

    print(
        f"Section {index}/{total_chunks} complete "
        f"({chunk_duration:.1f} seconds)",
        flush=True,
    )


# --------------------------------------------------
# PRINT ONLY THE FINAL TRANSCRIPT
# --------------------------------------------------
full_transcript = " ".join(
    text for text in all_transcripts if text
)

print("\n" + full_transcript)
