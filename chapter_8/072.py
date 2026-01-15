import torch
import torch.nn as nn
import importlib
import sys
import os

sys.path.insert(0, '/home/takeuchi/workspace/nlp_100_nocks/chapter_8')
module_070 = importlib.import_module('070')
module_071 = importlib.import_module('071')

load_word_embeddings = module_070.load_word_embeddings
load_and_process_sst = module_071.load_and_process_sst

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

    train_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/train.tsv'
    dev_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/dev.tsv'

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_072.txt')

    try:
        # 071.pyのデータセットを取得
        train_dataset = load_and_process_sst(train_file, w2id)
        dev_dataset = load_and_process_sst(dev_file, w2id)

        # モデルを構築
        model = SentimentClassifier(embedding_matrix=E)

        results = [
            "--- SentimentClassifier Model ---",
            f"モデルアーキテクチャ:\n{model}",
            "",
            "--- パラメータ情報 ---",
            f"Embedding層の語彙数: {E.shape[0]}",
            f"Embedding次元数: {E.shape[1]}",
            f"全パラメータ数: {sum(p.numel() for p in model.parameters())}",
            "",
            "--- データセット情報 ---",
            f"訓練データ数: {len(train_dataset)}",
            f"開発データ数: {len(dev_dataset)}"
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in results:
                print(line)
                f.write(line + '\n')
        
        print(f"\n結果を保存しました: {output_file}")

    except FileNotFoundError as e:
        error_msg = f"エラー: ファイルが見つかりません。\n{e}"
        print(error_msg)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)