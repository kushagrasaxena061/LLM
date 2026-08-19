import os
import subprocess
import sys

def install_and_import():
    try:
        import datasets
    except ImportError:
        print("📦 Installing Hugging Face 'datasets' library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        import datasets
    return datasets

datasets = install_and_import()

# 108 Million Words Total (Perfectly matches 70,000 steps)
TARGET_TOTAL_WORDS = 108_000_000
WORDS_PER_DATASET = TARGET_TOTAL_WORDS // 3

print(f"🚀 Streaming High-Quality Data (Target: {TARGET_TOTAL_WORDS:,} words)...")

def stream_to_file(repo, subset, target_words, f):
    print(f"📥 Streaming {repo} ({subset})...")
    word_count = 0
    try:
        ds = datasets.load_dataset(repo, subset, split="train", streaming=True)
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            
            words = len(text.split())
            f.write(text + "\n<|endoftext|>\n")
            word_count += words
            
            if word_count >= target_words:
                break
                
        print(f"✅ Extracted {word_count:,} words from {subset}.")
    except Exception as e:
        print(f"⚠️ Failed to stream {subset}: {e}")

with open("final_108M_corpus.txt", "w", encoding="utf-8") as f:
    stream_to_file("HuggingFaceFW/fineweb-edu", "sample-10BT", WORDS_PER_DATASET, f)
    stream_to_file("HuggingFaceTB/cosmopedia", "v2", WORDS_PER_DATASET, f)
    stream_to_file("HuggingFaceTB/smollm-corpus", "python-edu", WORDS_PER_DATASET, f)

size_mb = os.path.getsize("final_108M_corpus.txt") / (1024 * 1024)
print(f"\n🎉 'final_108M_corpus.txt' created successfully! Size: {size_mb:.2f} MB")
