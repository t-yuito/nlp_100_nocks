import os
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-cased", top_k=10)

text = "The movie was full of [MASK]."

results = fill_mask(text)

output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "082.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Sentence: {text}\n")
    f.write(f"-" * 30 + "\n")
    for i, res in enumerate(results, 1):
        token = res['token_str']
        score = res['score']
        line = f"{i:2d}: {token:<15} (Score: {score:.4f})"
        f.write(line + "\n")
        print(line)

print("実行完了")