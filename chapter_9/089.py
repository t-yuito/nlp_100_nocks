import os
import torch
import torch.nn as nn
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer
)

# 1. カスタムモデルの定義
class BERTMaxPollingModel(nn.Module):
    def __init__(self, model_ckpt, num_labels=2):
        super().__init__()
        # 内部で標準のBERTモデルを保持
        self.bert = AutoModel.from_pretrained(model_ckpt)
        # 分類用の線形層（BERTの隠れ層サイズ 768 -> クラス数 2）
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # BERTの出力を取得
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        
        # マスクを考慮してパディング部分を無視しつつ最大値をとる
        # 簡易的には torch.max(last_hidden_state, dim=1)[0]
        # ここでは「各次元で最も強い反応があったトークン」を抽出
        max_embeddings = torch.max(last_hidden_state, dim=1)[0] # [batch_size, hidden_size]
        
        logits = self.classifier(max_embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Trainerが期待する出力を返す
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# 2. セットアップ
model_ckpt = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
raw_datasets = load_dataset("glue", "sst2")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 3. カスタムモデルのインスタンス化
model = BERTMaxPollingModel(model_ckpt)

# 4. 評価指標と学習の設定
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results_maxpooling",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # メモリに余裕があるなら増やす
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none"
)

# 5. 学習の実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

print("--- カスタムアーキテクチャ（Max Pooling）での学習開始 ---")
trainer.train()

# 6. 結果の評価と保存
eval_results = trainer.evaluate()
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "089.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Validation Accuracy (Max Pooling): {eval_results['eval_accuracy']:.4f}\n")

print(f"\n検証セット上での正解率: {eval_results['eval_accuracy']:.4f}")