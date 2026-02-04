# Chapter 9: Transformers / BERT

## 概要
Hugging Face Transformersライブラリを使った事前学習済みBERTモデルの活用問題です。トークナイゼーション、マスク言語モデル、文埋め込み、ファインチューニングなど、BERTの様々な使い方を学びます。

## 問題一覧

| 問題番号 | タイトル | 内容 |
|----------|----------|------|
| 80 | サブワード分割 | BERTトークナイザによる単語分割（WordPiece） |
| 81 | 穴埋め予測（トップ1） | マスク言語モデルで[MASK]に入る単語を予測 |
| 82 | 穴埋め予測（トップ10） | [MASK]に入る候補単語を上位10件取得 |
| 83 | 文ベクトル（CLSトークン） | [CLS]トークンを使った文の埋め込みと類似度計算 |
| 84 | 文ベクトル（平均プーリング） | 全トークンの平均ベクトルによる類似度計算 |
| 85 | SST-2データセットの読み込み | GLUEベンチマークのSST-2をトークナイズ |
| 86 | ミニバッチ化とパディング | 複数文をパディングでバッチ化 |
| 87 | ファインチューニング | SST-2データセットでBERTを感情分析用に学習 |
| 88 | 感情予測 | 学習済みモデルで任意テキストの感情を予測 |
| 89 | Max Poolingによるファインチューニング | カスタムプーリング層を使ったモデルの実装 |

## 主要な概念と技術

### 1. BERTトークナイザ
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer.tokenize("incomprehensibilities")
# → ['in', '##com', '##p', '##re', '##hen', '##si', '##bili', '##ties']
```

### 2. Mask言語モデル（Fill-Mask）
```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-cased")
results = fill_mask("The movie was full of [MASK].")
```

### 3. 文ベクトルの取得
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
outputs = model(**inputs)

# CLSトークン
cls_embedding = outputs.last_hidden_state[0, 0, :]

# 平均プーリング
mean_embedding = torch.mean(outputs.last_hidden_state[0], dim=0)
```

### 4. Trainerによるファインチューニング
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()
```

## 学習ポイント
- Hugging Face Transformersの基本的な使い方
- サブワードトークナイゼーション（WordPiece）
- 事前学習済みモデルの読み込みと活用
- `pipeline` APIによる簡易推論
- 文埋め込みの取得方法（CLS / Mean Pooling / Max Pooling）
- `Trainer` APIによるファインチューニング
- カスタムモデルの実装（`nn.Module`継承）
- `evaluate`ライブラリによる評価指標計算

## 使用ライブラリ
- `transformers` - BERTモデル・トークナイザ
- `datasets` - GLUEデータセットの読み込み
- `evaluate` - 評価指標（Accuracy）
- `torch` - テンソル操作・カスタムモデル
- `scikit-learn` - コサイン類似度計算

## 実行方法

```bash
# 個別実行
uv run 080.py

# GPU指定で学習（087, 089）
CUDA_VISIBLE_DEVICES=0 uv run 087.py
```
