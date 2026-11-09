
from pathlib import Path
import time
import sys

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_DIR = Path(
    r"D:\AI_Voice\models\bangladeshi_bangla_tts_vits"
)

MODEL_PATH = MODEL_DIR / "pytorch_model.pth"
CONFIG_PATH = MODEL_DIR / "config.json"

OUTPUT_DIR = Path(r"D:\AI_Voice\output")
OUTPUT_FILE = OUTPUT_DIR / "bangla_vits_test.wav"

# Approximately 40 Bengali words
BENGALI_TEXT = """
আজ সকালে আমি একটি নতুন প্রযুক্তি নিয়ে কাজ শুরু করেছি।
এই প্রযুক্তির মাধ্যমে বাংলা ভাষার লেখা থেকে স্বাভাবিক কণ্ঠে
অডিও তৈরি করা সম্ভব। আমাদের লক্ষ্য হলো বাংলাদেশের মানুষের
কথা বলার ধরন, উচ্চারণ এবং ভাষার স্বাভাবিক সৌন্দর্য বজায় রাখা।
ভবিষ্যতে এই প্রযুক্তি ব্যবহার করে ভিডিও, গল্প, শিক্ষামূলক
কনটেন্ট এবং বিভিন্ন ডিজিটাল প্রকল্পের জন্য উন্নত মানের
বাংলা কণ্ঠ তৈরি করা যাবে।
""".strip()


def show_step(message):
    print(f"\n[STEP] {message}", flush=True)


def main():
    print("=" * 58)
    print("       BANGLADESHI BANGLA TTS - VITS TEST")
    print("=" * 58)

    # Step 1: Check files
    show_step("Checking local model files...")

    if not MODEL_PATH.is_file():
        print(f"[ERROR] Model weights not found:\n{MODEL_PATH}")
        return 1

    if not CONFIG_PATH.is_file():
        print(f"[ERROR] Config not found:\n{CONFIG_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[OK] Model weights found.")
    print("[OK] Configuration found.")

    # Step 2: Show text
    show_step("Preparing Bengali text...")
    print("\nText to convert:\n")
    print(BENGALI_TEXT)

    # Step 3: Load model
    show_step("Loading VITS model on CPU...")
    print("The first load may take some time.", flush=True)

    start_load = time.perf_counter()

    try:
        from TTS.utils.synthesizer import Synthesizer

        synthesizer = Synthesizer(
            tts_checkpoint=str(MODEL_PATH),
            tts_config_path=str(CONFIG_PATH),
            use_cuda=False,
        )

    except Exception as exc:
        print(f"\n[ERROR] Model loading failed:\n{exc}")
        return 1

    load_time = time.perf_counter() - start_load
    print(f"[OK] Model loaded in {load_time:.2f} seconds.")

    # Step 4: Generate speech
    show_step("Converting Bengali text into speech...")
    start_generation = time.perf_counter()

    try:
        audio = synthesizer.tts(BENGALI_TEXT)

        print("[OK] Speech generation completed.")

        # Step 5: Save WAV
        show_step("Saving generated audio...")
        synthesizer.save_wav(audio, str(OUTPUT_FILE))

    except Exception as exc:
        print(f"\n[ERROR] Speech generation failed:\n{exc}")
        return 1

    generation_time = time.perf_counter() - start_generation

    # Step 6: Report results
    print("\n" + "=" * 58)
    print("                 TEST COMPLETED")
    print("=" * 58)
    print(f"Model load time : {load_time:.2f} seconds")
    print(f"Generation time : {generation_time:.2f} seconds")
    print(f"Output file     : {OUTPUT_FILE}")
    print(f"File size       : {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print("=" * 58)
    print("\nOpen the WAV file and listen to the Bengali pronunciation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
