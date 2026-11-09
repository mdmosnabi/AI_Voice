from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

model_dir = r"D:\AI_Voice\models\banglaasr\models--bangla-speech-processing--BanglaASR\snapshots\89c3a1b9585be48d9527cd8964c150fe2eebdbaa"

print("Model directory:", model_dir)
print("Preprocessor exists:", os.path.isfile(os.path.join(model_dir, "preprocessor_config.json")))

processor = WhisperProcessor.from_pretrained(
    model_dir,
    local_files_only=True
)

model = WhisperForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True
)

print("Processor loaded:", type(processor).__name__)
print("Tokenizer:", type(processor.tokenizer).__name__)
print("Generation config:")
print(model.generation_config)
