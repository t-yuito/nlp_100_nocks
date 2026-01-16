import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import importlib
import sys
import os

sys.path.insert(0, '/home/takeuchi/workspace/nlp_100_nocks/chapter_8')
module_070 = importlib.import_module('070')
module_071 = importlib.import_module('071')
module_075 = importlib.import_module('075')

load_word_embeddings = module_070.load_word_embeddings
load_and_process_sst = module_071.load_and_process_sst
collate_fn = module_075.collate_fn


# アーキテクチャ1: 多層ニューラルネットワーク（MLP）
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, output_dim=1, dropout=0.3):
        super().__init__()
        
        vocab_size, embed_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # 多層ニューラルネットワーク
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids):
        # Embedding
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # 平均プーリング
        mean_embeds = embeds.mean(dim=1)  # (batch_size, embed_dim)
        
        # 多層FC
        hidden = self.relu(self.fc1(mean_embeds))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        logits = self.fc3(hidden)
        
        return logits


# アーキテクチャ2: LSTM
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, output_dim=1, n_layers=2, dropout=0.3):
        super().__init__()
        
        vocab_size, embed_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, seq_lengths=None):
        # Embedding
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # LSTM
        if seq_lengths is not None:
            # パッキング（パディングを無視）
            packed_embeds = pack_padded_sequence(
                embeds, 
                seq_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embeds)
            # アンパッキング
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embeds)
        
        # 最後のhidden stateを使用
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        
        logits = self.fc(last_hidden)
        
        return logits


def custom_collate_fn(batch):
    """シーケンス長情報を含むcollate関数"""
    batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
    
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['label'] for item in batch]
    
    # シーケンス長を記録
    seq_lengths = torch.tensor([len(ids) for ids in input_ids_list])
    
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    stacked_labels = torch.stack(labels_list)
    
    return {
        'input_ids': padded_input_ids,
        'label': stacked_labels,
        'seq_lengths': seq_lengths
    }


if __name__ == '__main__':
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # パラメータ設定
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001  # 小さめに設定
    NUM_EPOCHS = 10
    
    # ファイルパスとディレクトリ設定
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/GoogleNews-vectors-negative300.bin.gz'
    train_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/train.tsv'
    dev_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/dev.tsv'

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_079.txt')

    try:
        # データセット読み込み
        print("データセット読み込み中...")
        E, w2id, id2w = load_word_embeddings(model_path, limit=300000)
        train_dataset = load_and_process_sst(train_file, w2id)
        dev_dataset = load_and_process_sst(dev_file, w2id)

        print(f"訓練データ数: {len(train_dataset)}")
        print(f"開発データ数: {len(dev_dataset)}")

        # データローダーの作成
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=custom_collate_fn
        )

        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        log_results = ["=== 79. アーキテクチャの変更（LSTM実装） ===\n"]

        # 複数のアーキテクチャを試す
        architectures = [
            ("MultiLayerPerceptron", MultiLayerPerceptron(E, hidden_dim=256)),
            ("LSTM", LSTMSentimentClassifier(E, hidden_dim=256, n_layers=2))
        ]

        for arch_name, model in architectures:
            print(f"\n{'='*50}")
            print(f"アーキテクチャ: {arch_name}")
            print(f"{'='*50}")
            
            log_results.append(f"\n=== {arch_name} ===\n")
            
            # パラメータ数を表示
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            log_results.append(f"パラメータ数: {total_params}")
            log_results.append(f"更新対象パラメータ数: {trainable_params}\n")
            
            model.to(device)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            # 学習ループ
            for epoch in range(NUM_EPOCHS):
                model.train()
                total_loss = 0
                batch_count = 0
                
                for batch in train_dataloader:
                    inputs = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    seq_lengths = batch['seq_lengths'].to(device)
                    
                    optimizer.zero_grad()
                    
                    if arch_name == "LSTM":
                        outputs = model(inputs, seq_lengths)
                    else:
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
            
            # 開発セットの正解率計算
            model.eval()
            dev_correct = 0
            dev_total = 0
            
            with torch.no_grad():
                for batch in dev_dataloader:
                    inputs = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    seq_lengths = batch['seq_lengths'].to(device)
                    
                    if arch_name == "LSTM":
                        outputs = model(inputs, seq_lengths)
                    else:
                        outputs = model(inputs)
                    
                    predicted = (outputs > 0).float()
                    
                    dev_total += labels.size(0)
                    dev_correct += (predicted == labels).sum().item()
            
            dev_acc = dev_correct / dev_total
            dev_msg = f"\n開発セット正解率: {dev_acc:.4f} ({dev_correct}/{dev_total})\n"
            print(dev_msg)
            log_results.append(dev_msg)

        # 結果をファイルに保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in log_results:
                f.write(line + '\n')

        print(f"\n結果を保存しました: {output_file}")

    except Exception as e:
        import traceback
        error_msg = f"エラーが発生しました: {e}\n{traceback.format_exc()}"
        print(error_msg)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)
