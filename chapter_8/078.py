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
module_075 = importlib.import_module('075')

load_word_embeddings = module_070.load_word_embeddings
load_and_process_sst = module_071.load_and_process_sst
SentimentClassifier = module_072.SentimentClassifier
collate_fn = module_075.collate_fn


if __name__ == '__main__':
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    if torch.cuda.is_available():
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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
    output_file = os.path.join(output_dir, 'result_078.txt')

    try:
        # データセット読み込み
        print("データセット読み込み中...")
        E, w2id, id2w = load_word_embeddings(model_path, limit=300000)
        train_dataset = load_and_process_sst(train_file, w2id)
        dev_dataset = load_and_process_sst(dev_file, w2id)

        print(f"訓練データ数: {len(train_dataset)}")
        print(f"開発データ数: {len(dev_dataset)}")

        # データローダーの作成（問題75のcollate_fnを使用）
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        # モデルの初期化
        model = SentimentClassifier(embedding_matrix=E)
        
        # 【問題78】単語埋め込みパラメータも更新する（ファインチューニング）
        # requires_grad = Trueのままにする（デフォルト値）
        # model.embedding.weight.requires_grad = False  # ← この行を削除
        
        # モデルをGPUに移動
        model.to(device)

        # 損失関数とオプティマイザ
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # 学習ループ
        log_results = ["=== 78. 単語埋め込みのファインチューニング ===\n"]
        log_results.append(f"デバイス: {device}")
        if torch.cuda.is_available():
            log_results.append(f"GPU名: {torch.cuda.get_device_name(0)}")
        log_results.append(f"Batch Size: {BATCH_SIZE}")
        log_results.append(f"Learning Rate: {LEARNING_RATE}")
        log_results.append(f"Num Epochs: {NUM_EPOCHS}")
        log_results.append(f"単語埋め込みパラメータ: ファインチューニング有効\n")

        # パラメータ情報を表示
        embedding_params = model.embedding.weight.numel()
        fc_params = sum(p.numel() for p in model.fc.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        
        log_results.append("=== パラメータ情報 ===")
        log_results.append(f"Embedding層パラメータ数: {embedding_params}")
        log_results.append(f"FC層パラメータ数: {fc_params}")
        log_results.append(f"全パラメータ数: {total_params}")
        log_results.append(f"更新対象パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

        print("学習を開始します。")
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            batch_count = 0

            for batch in train_dataloader:
                # テンソルをGPUに移動
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            epoch_msg = f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}"
            print(epoch_msg)
            log_results.append(epoch_msg)

        print("学習完了\n")
        log_results.append("\n=== 開発セットにおける正解率 ===\n")

        # 開発セットの正解率計算
        model.eval()
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for batch in dev_dataloader:
                # テンソルをGPUに移動
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(inputs)
                
                # ロジットが0より大きければ1（Positive）、それ以外は0（Negative）
                predicted = (outputs > 0).float()
                
                dev_total += labels.size(0)
                dev_correct += (predicted == labels).sum().item()

        dev_acc = dev_correct / dev_total
        dev_msg = f"開発セット正解率: {dev_acc:.4f} ({dev_correct}/{dev_total})"
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
