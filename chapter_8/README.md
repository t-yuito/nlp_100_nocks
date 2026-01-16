# Chapter 8: ニューラルネットワークを用いた文書分類

## 概要
PyTorchを使ったニューラルネットワークの実装と学習を学びます。感情分析タスク（SST-2）を題材に、モデル設計から学習、評価、最適化までの一連のプロセスを実践します。

## 問題一覧

| 問題番号 | タイトル | 内容 | 主要概念 |
|----------|----------|------|---------|
| 70 | 単語埋め込みの読み込み | Word2Vecモデルのロード、埋め込み行列の構築 | `KeyedVectors`, 単語↔ID マッピング |
| 71 | データセット読み込み | SST-2データセットの処理、テキスト→ID変換 | `pd.iterrows()`, リスト内包表記 |
| 72 | モデル構築 | `nn.Module`を使ったニューラルネットワークモデル | Embedding層、線形層、forward() |
| 73 | 学習 | ミニバッチ学習の基本実装 | DataLoader、損失関数、オプティマイザ |
| 74 | 評価 | 学習済みモデルの精度計算 | `model.eval()`, 混同行列、評価指標 |
| 75 | パディング | バッチ処理でのパディング実装 | `pad_sequence`, 系列長ソート |
| 76 | ミニバッチ学習 | パディング処理を活用した効率的な学習 | バッチ処理、正解率計算 |
| 77 | GPU上での学習 | GPU加速による高速学習 | `device`, `.to(device)`, CUDA |
| 78 | 単語埋め込みのファインチューニング | 埋め込み層の重みも同時に更新 | `requires_grad`, パラメータ更新 |
| 79 | アーキテクチャの変更 | 複数のモデルアーキテクチャの実装・比較 | MLP, LSTM, pack_padded_sequence |

## 主要な概念と技術

### 1. データ処理パイプライン
```
Word2Vecモデル
    ↓
SST-2データセット（テキスト）
    ↓
トークン化 → ID変換
    ↓
テンソル化（input_ids, labels）
```

### 2. モデル構築（PyTorch）
```python
class SentimentClassifier(nn.Module):
    - Embedding層：ID → ベクトル変換
    - 線形層（FC層）：分類用
    - forward()：順伝播の定義
```

### 3. 学習フロー
```
1. データローダーでミニバッチ取得
2. 順伝播 → 損失計算
3. 逆伝播 → 勾配計算
4. パラメータ更新
```

### 4. 評価指標
- **正解率（Accuracy）**：全体的な精度
- **混同行列（Confusion Matrix）**：予測の内訳
- **適合率・再現率・F1スコア**：クラス間の詳細評価

## 実装のポイント

### パディング処理（問題75）
```python
# 系列長でソート（降順）
batch.sort(key=lambda x: len(x['input_ids']), reverse=True)

# 最大長に揃える
padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
```
→ 効率的なバッチ処理、メモリ最適化

### GPU活用（問題77）
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
inputs = batch['input_ids'].to(device)
```
→ 高速な学習、大規模データ対応

### ファインチューニング（問題78）
```python
# 埋め込み層も更新
model.embedding.weight.requires_grad = True
```
→ 事前学習モデルのタスク特化

### 複数アーキテクチャ（問題79）

**MLP（多層パーセプトロン）**
```
Embedding → 平均プーリング → FC層 × 3 → 出力
```
シンプルで高速

**LSTM（再帰型ニューラルネットワーク）**
```
Embedding → LSTM層 × 2 → 最後のhidden → FC層 → 出力
```
系列情報を活用、より高精度

## 学習ポイント

| トピック | 内容 |
|---------|------|
| **テンソル操作** | `shape`, `unsqueeze()`, `stack()`, `cat()` |
| **モデル実装** | `nn.Module`, `nn.Embedding`, `nn.Linear`, `nn.LSTM` |
| **学習** | 損失関数、オプティマイザ（SGD, Adam）、逆伝播 |
| **評価** | `model.eval()`, `torch.no_grad()`, 評価指標 |
| **効率化** | DataLoader, pad_sequence, pack_padded_sequence |
| **GPU** | CUDA, device管理、メモリ効率 |
| **正則化** | Dropout, weight decay |

## データセット

**SST-2（Stanford Sentiment Treebank）**
- 映画レビューのテキスト分類
- ラベル：0（ネガティブ）, 1（ポジティブ）
- 訓練データ：約67,000件
- 開発データ：約873件

**Word2Vec（Google News）**
- 300次元の事前学習単語ベクトル
- 300,000個の単語

## 実行方法

```bash
# 順番に実行することで学習のプロセスを体験
cd /home/takeuchi/workspace/nlp_100_nocks/chapter_8

uv run 070.py  # 埋め込みの読み込み
uv run 071.py  # データセット処理
uv run 072.py  # モデル構築
uv run 073.py  # 基本学習
uv run 074.py  # 評価
uv run 075.py  # パディング検証
uv run 076.py  # ミニバッチ学習
uv run 077.py  # GPU学習
uv run 078.py  # ファインチューニング
uv run 079.py  # 複数アーキテクチャ
```

## 出力ファイル

全ての結果は `/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out/` に保存されます：

- `result_070.txt` ～ `result_079.txt`：各問題の実行結果
- 学習曲線、正解率、パラメータ情報などを記録

## 応用・発展

1. **他のアーキテクチャ**
   - CNN（畳み込みニューラルネットワーク）
   - Transformer, BERT
   - Attention機構

2. **データ拡張**
   - 訓練データの前処理
   - クラスバランスの調整
   - 不均衡データの対策

3. **ハイパーパラメータチューニング**
   - 学習率、バッチサイズ
   - エポック数、隠れ層次元
   - Dropout率

4. **他のタスク**
   - 固有表現認識（NER）
   - 質問応答（QA）
   - テキスト要約

## まとめ

Chapter 8を通じて、以下の一連のプロセスを習得できます：

1. **データ準備**：テキスト → テンソル化
2. **モデル設計**：アーキテクチャの選択・実装
3. **学習**：勾配降下法、バッチ処理
4. **評価**：各種指標による性能評価
5. **最適化**：GPU活用、ファインチューニング
6. **改善**：アーキテクチャ変更による精度向上

これらの知識は、自然言語処理だけでなく、深層学習全般に応用可能です。
