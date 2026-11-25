
from pathlib import Path
import sys
import time
import wave

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "indic-tiny-tts-bengali"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))
from indic_tiny_tts import TinyTTS

VOICE = "male"
SPEED = 1.05

TEXT = """
একটি অদ্ভুত ঘটনা দিয়ে শুরু হয় এই সিনেমার গল্প।

একদিন সকালে ঘুম থেকে উঠে লোকটি দেখতে পেল,
তার চারপাশের পৃথিবী সম্পূর্ণ বদলে গেছে।
কিন্তু সবচেয়ে অবাক করার বিষয় হলো,
কেউ যেন তাকে চিনতেই পারছে না।

সে বুঝতে পারছিল না আসলে কী ঘটেছে।
প্রতিটি মুহূর্তে তার সামনে নতুন নতুন রহস্য
এবং বিপদের দরজা খুলে যাচ্ছিল।

অন্যদিকে, শহরের মানুষজন তখনও স্বাভাবিক জীবনযাপন করছিল।
কেউ অফিসে যাচ্ছে, কেউ বাজার করছে,
আবার কেউ নিজের পরিবারের সঙ্গে সময় কাটাচ্ছে।

কিন্তু এই স্বাভাবিক পরিবেশের আড়ালে
লুকিয়ে ছিল এক ভয়ংকর সত্য।
আর সেই সত্য জানার জন্য
তাকে পাড়ি দিতে হবে এক কঠিন পথ।

হঠাৎ করেই তার সামনে হাজির হয়
এক রহস্যময় ব্যক্তি।
লোকটি তাকে এমন কিছু তথ্য জানায়,
যা তার জীবনের সব হিসাব বদলে দেয়।

এখন প্রশ্ন হলো, সে কি সত্যিই
এই রহস্যের সমাধান করতে পারবে?
নাকি তার জন্য অপেক্ষা করছে আরও বড় কোনো বিপদ?

সিনেমার পরবর্তী ঘটনাগুলো
আপনাকে এক মুহূর্তের জন্যও
চোখ সরাতে দেবে না।
"""

def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}")

    print("Loading Indic Tiny TTS...")
    start = time.perf_counter()

    tts = TinyTTS.from_pretrained(
        str(MODEL_DIR),
        device="cpu",
        threads=4,
    )

    print(f"Model loaded in {time.perf_counter() - start:.2f}s")

    print(f"\nGenerating Male Voice at {SPEED}x...")
    start = time.perf_counter()

    wav = tts.synthesize(
        TEXT,
        voice=VOICE,
        speed=SPEED,
    )

    generation_time = time.perf_counter() - start

    output_file = OUTPUT_DIR / "male_speed_1.15x.wav"
    tts.save_wav(wav, str(output_file))

    with wave.open(str(output_file), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()

    print("\nGeneration Completed!")
    print(f"Generation Time : {generation_time:.2f}s")
    print(f"Audio Duration  : {duration:.2f}s")
    print(f"Saved File      : {output_file.resolve()}")

if __name__ == "__main__":
    main()
