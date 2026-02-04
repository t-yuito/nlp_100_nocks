import os
from datasets import load_dataset
from transformers import AutoTokenizer

model_ckpt = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

raw_datasets = load_dataset("glue", "sst2")

def tokenize_function(examples):
    # tokenizer.tokenize()は、単語をバラバラの文字列に分解するだけの処理。
    return {"tokens": [tokenizer.tokenize(text) for text in examples["sentence"]]}

tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)

output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "085.txt")

with open (output_path, "w", encoding='utf-8') as f:
    for split in ["train", "validation"]:
        f.write(f"--- {split} set (first 3 examples) ---\n")
        for i in range(3):
            data = tokenized_dataset[split][i]
            line = f"Label: {data['label']} | Tokens: {data['tokens']}\n"
            f.write(line)
            print(f"[{split}] {line.strip()}")
        f.write("\n")

print("処理完了")