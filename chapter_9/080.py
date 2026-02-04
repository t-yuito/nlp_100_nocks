import os
from transformers import AutoTokenizer
from datasets import Dataset

model_ckpt = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
target_text = "The movie was full of incomprehensibilities."

ds = Dataset.from_dict({"text": [target_text]})

def tokenize_function(examples):
    return {"tokens": [tokenizer.tokenize(t) for t in examples["text"]]}

tokenized_ds = ds.map(tokenize_function, batched=True)
tokens = tokenized_ds[0]["tokens"]

output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "080.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(str(tokens))

print(f"保存内容：{tokens}")