import os
from datasets import load_dataset
from transformers import AutoTokenizer

model_ckpt = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
raw_datasets = load_dataset("glue", "sst2")

sample_sentences = raw_datasets["train"]["sentence"][:4]

# padding処理を行いミニバッチを作成
# return_tensors="pt" でPyTorchテンソル形式にする。
batch_inputs = tokenizer(
    sample_sentences,
    padding=True,
    truncation=True, # 限界を超えたら切り捨てる
    return_tensors="pt"
)

output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "086.txt")

with open(output_path, "w", encoding='utf-8') as f:
    f.write("--- Mini-batch Info ---\n")
    f.write(f"Shape of input_ids: {batch_inputs['input_ids'].shape}\n\n")

    for i, tokens in enumerate(batch_inputs["input_ids"]):
        decoded_tokens = tokenizer.convert_ids_to_tokens(tokens)
        line = f"Sentence {i}: {decoded_tokens}\n"
        f.write(line)
        print(line.strip())

print("ミニバッチ作成完了")