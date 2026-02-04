import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

model_ckpt = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

embeddings = []
for text in sentences:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # 全トークンの最終層ベクトルを取得
    # outputs.last_hidden_stateの形状: [batch_size, sequence_length, hidden_size]
    last_hidden_state = outputs.last_hidden_state[0] # [sequence_length, hidden_size]

    # 平均を取る。
    mean_embedding = torch.mean(last_hidden_state, dim=0).numpy()
    embeddings.append(mean_embedding)

matrix = cosine_similarity(embeddings)
    
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "084.txt")

with open(output_path, "w", encoding='utf-8') as f:
    f.write("Cosine Similarity Matrix (Mean Pooling):\n")
    header = "     " + "".join(f"Sent {i} " for i in range(len(sentences)))
    f.write(header + "\n")
    print(header)

    for i, row in enumerate(matrix):
        line = f"Sent {i}: " + "  ".join([f"{val:.4f}" for val in row])
        f.write(line + "\n")
        print(line)

print("保存完了")