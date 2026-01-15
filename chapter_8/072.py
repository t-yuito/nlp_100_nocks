import torch
import torch.nn as nn
import importlib
import sys
import os

sys.path.insert(0, '/home/takeuchi/workspace/nlp_100_nocks/chapter_8')
module = importlib.import_module('070')
load_word_embeddings = module.load_word_embeddings

class SentimentClassifier(nn.Module):
    def __init__(self, embedding_matrix, output_dim=1):
        super().__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)

        mean_embeds = embeds.mean(dim=1)

        logits = self.fc(mean_embeds)

        return logits


if __name__ == '__main__':
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/GoogleNews-vectors-negative300.bin.gz'
    E, w2id, id2w = load_word_embeddings(model_path, limit=300000)

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_072.txt')

    model = SentimentClassifier(embedding_matrix=E)

    results = [
        "--- SentimentClassifier Model ---",
        f"モデルアーキテクチャ:\n{model}",
        "",
        "--- パラメータ情報 ---",
        f"Embedding層の語彙数: {E.shape[0]}",
        f"Embedding次元数: {E.shape[1]}",
        f"全パラメータ数: {sum(p.numel() for p in model.parameters())}"
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            print(line)
            f.write(line + '\n')
    
    print(f"\n結果を保存しました: {output_file}")