import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import importlib
import sys
import os

sys.path.insert(0, '/home/takeuchi/workspace/nlp_100_nocks/chapter_8')
module_070 = importlib.import_module('070')
module_071 = importlib.import_module('071')
module_072 = importlib.import_module('072')
module_073 = importlib.import_module('073')

load_word_embeddings = module_070.load_word_embeddings
load_and_process_sst = module_071.load_and_process_sst
SentimentClassifier = module_072.SentimentClassifier
collate_fn = module_073.collate_fn


def calculate_accuracy(model, dataset, batch_size=32):
    # 評価用データローダーの作成
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    model.eval()  # 評価モード
    
    correct = 0
    total = 0
    
    with torch.no_grad():  # 勾配計算不要
        for inputs, labels in loader:
            outputs = model(inputs)
            
            # ロジットが0より大きければ1（Positive）、それ以外は0（Negative）と予測
            # labelsと同じ形状にする
            predicted = (outputs > 0).float()
            
            # 正解数をカウント
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total


if __name__ == '__main__':
    # パラメータ設定
    BATCH_SIZE = 32
    LEARNING_RATE = 0.5
    NUM_EPOCHS = 10

    # ファイルパスとディレクトリ設定
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/GoogleNews-vectors-negative300.bin.gz'
    train_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/train.tsv'
    dev_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/dev.tsv'

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_074.txt')

    try:
        # データセット読み込み
        print("データセット読み込み中...")
        E, w2id, id2w = load_word_embeddings(model_path, limit=300000)
        train_dataset = load_and_process_sst(train_file, w2id)
        dev_dataset = load_and_process_sst(dev_file, w2id)

        # データローダーの作成
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        # モデルの初期化
        model = SentimentClassifier(embedding_matrix=E)
        model.embedding.weight.requires_grad = False

        # 損失関数とオプティマイザ
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # 学習ループ
        log_results = ["=== 学習フェーズ ===\n"]

        print("学習を開始します。")
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0

            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            epoch_msg = f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}"
            print(epoch_msg)
            log_results.append(epoch_msg)

        print("学習完了\n")
        log_results.append("\n=== 評価フェーズ ===\n")

        # --- 訓練データの正解率（参考）---
        train_acc = calculate_accuracy(model, train_dataset)
        train_msg = f"訓練データの正解率: {train_acc:.4f}"
        print(train_msg)
        log_results.append(train_msg)

        # --- 開発データの正解率（本題）---
        dev_acc = calculate_accuracy(model, dev_dataset)
        dev_msg = f"開発データの正解率: {dev_acc:.4f}"
        print(dev_msg)
        log_results.append(dev_msg)

        # 結果をファイルに保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in log_results:
                f.write(line + '\n')

        print(f"\n結果を保存しました: {output_file}")

    except Exception as e:
        error_msg = f"エラーが発生しました: {e}"
        print(error_msg)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)