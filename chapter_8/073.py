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

load_word_embeddings = module_070.load_word_embeddings
load_and_process_sst = module_071.load_and_process_sst
SentimentClassifier = module_072.SentimentClassifier

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    # input_idsを、長さを合わせるためにpadding
    # batch_first=True: 出力の形状を (batch_size, seq_len) にする
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)

    labels = torch.stack(labels)

    return input_ids_padded, labels


if __name__ == '__main__':
    BATCH_SIZE = 32
    LEARNING_RATE = 0.5
    NUM_EPOCHS = 10

    # ファイルパスとディレクトリ設定
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/GoogleNews-vectors-negative300.bin.gz'
    train_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/train.tsv'
    dev_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/dev.tsv'

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_073.txt')

    try:
        # データセット読み込み
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

        # 単語埋め込みパラメータの固定。勾配計算を無効化
        model.embedding.weight.requires_grad = False

        # 損失関数とオプティマイザ
        criterion = nn.BCEWithLogitsLoss() # シグモイド関数 + バイナリ交差エントロピー
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # 学習ループ
        log_results = ["学習を開始します。\n"]

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0

            for i, (inputs, labels) in enumerate(train_dataloader):
                # 勾配の初期化
                optimizer.zero_grad()

                # 順伝播
                outputs = model(inputs)

                # 損失関数
                loss = criterion(outputs, labels)

                # 逆伝播
                loss.backward()

                # パラメータ更新
                optimizer.step()

                total_loss += loss.item()

            # エポックごとの平均損失を表示
            avg_loss = total_loss / len(train_dataloader)
            epoch_msg = f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}"
            print(epoch_msg)
            log_results.append(epoch_msg)

        log_results.append("\n学習完了")

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