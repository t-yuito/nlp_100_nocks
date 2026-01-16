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
    """
    DataLoaderで使用するカスタムcollate関数
    Args:
        batch (list): データセットから取り出された辞書のリスト
                      [{'text': str, 'label': tensor, 'input_ids': tensor}, ...]
    Returns:
        dict: バッチ化されたデータ
              {'input_ids': tensor, 'label': tensor}
    """
    # input_idsの長さでソート
    batch.sort(key=lambda x: len(x['input_ids']), reverse=True) # 降順

    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['label'] for item in batch]

    # batch_first=True batch_sizeを先にして、seq_lenを後にする。
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)

    stacked_labels = torch.stack(labels_list)

    return {
        'input_ids': padded_input_ids,
        'label': stacked_labels
    }


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
    output_file = os.path.join(output_dir, 'result_075.txt')

    try:
        # データセット読み込み
        print("データセット読み込み中...")
        E, w2id, id2w = load_word_embeddings(model_path, limit=300000)
        train_dataset = load_and_process_sst(train_file, w2id)
        dev_dataset = load_and_process_sst(dev_file, w2id)

        # データローダーの作成（新しいcollate_fn使用）
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
        model.embedding.weight.requires_grad = False

        # 損失関数とオプティマイザ
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # 学習ループ
        log_results = ["=== 新しいcollate_fn（長さソート）での学習 ===\n"]

        print("学習を開始します。")
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0

            for batch in train_dataloader:
                inputs = batch['input_ids']
                labels = batch['label']

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

        # 評価フェーズ
        model.eval()
        
        # 訓練データの正解率
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for batch in train_dataloader:
                inputs = batch['input_ids']
                labels = batch['label']
                outputs = model(inputs)
                predicted = (outputs > 0).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total
        train_msg = f"訓練データの正解率: {train_acc:.4f}"
        print(train_msg)
        log_results.append(train_msg)

        # 開発データの正解率
        dev_correct = 0
        dev_total = 0
        with torch.no_grad():
            for batch in dev_dataloader:
                inputs = batch['input_ids']
                labels = batch['label']
                outputs = model(inputs)
                predicted = (outputs > 0).float()
                dev_total += labels.size(0)
                dev_correct += (predicted == labels).sum().item()

        dev_acc = dev_correct / dev_total
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