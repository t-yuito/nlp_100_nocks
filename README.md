# 言語処理100本ノック

[言語処理100本ノック](https://nlp100.github.io/ja/)のPython実装リポジトリです。

## 概要

自然言語処理（NLP）の基礎から応用までを段階的に学ぶための演習問題集です。文字列操作から始まり、形態素解析、機械学習、ニューラルネットワーク、BERTまでを網羅しています。

## 章構成

| 章 | タイトル | 問題番号 | 主な内容 |
|----|----------|----------|----------|
| [Chapter 1](chapter_1/) | 準備運動 | 00-09 | 文字列操作、リスト・辞書・集合の操作 |
| [Chapter 2](chapter_2/) | UNIXコマンド | 10-19 | ファイル操作、テキスト処理 |
| [Chapter 3](chapter_3/) | 正規表現 | 20-29 | 正規表現、Wikipedia記事の解析 |
| [Chapter 4](chapter_4/) | 形態素解析 | 30-39 | spaCy/GiNZA、係り受け解析、TF-IDF |
| [Chapter 5](chapter_5/) | 大規模言語モデル | 40-49 | Google Gemini API、プロンプトエンジニアリング |
| [Chapter 6](chapter_6/) | 単語ベクトル | 50-59 | Word2Vec、クラスタリング、t-SNE |
| [Chapter 7](chapter_7/) | 機械学習 | 60-69 | ロジスティック回帰、BoW、評価指標 |
| [Chapter 8](chapter_8/) | ニューラルネットワーク | 70-79 | PyTorch、Embedding、LSTM |
| [Chapter 9](chapter_9/) | Transformers | 80-89 | BERT、Hugging Face Transformers |

## 使用技術・ライブラリ

- **言語**: Python 3.12+
- **パッケージ管理**: uv
- **主要ライブラリ**:
  - `pandas`, `numpy` - データ処理
  - `spacy`, `ginza` - 形態素解析
  - `scikit-learn` - 機械学習
  - `gensim` - 単語ベクトル
  - `pytorch` - ニューラルネットワーク
  - `transformers` - BERT等のTransformerモデル
  - `matplotlib` - 可視化

## 環境構築

各章のディレクトリに`pyproject.toml`があります。

```bash
cd chapter_X
uv sync
uv run main.py  # または uv run 0XX.py
```

## ディレクトリ構成

```
nlp_100_nocks/
├── README.md
├── chapter_1/          # 準備運動
├── chapter_2/          # UNIXコマンド
├── chapter_3/          # 正規表現
├── chapter_4/          # 形態素解析
├── chapter_5/          # 大規模言語モデル
├── chapter_6/          # 単語ベクトル
├── chapter_7/          # 機械学習
├── chapter_8/          # ニューラルネットワーク
└── chapter_9/          # Transformers
```

## 参考リンク

- [言語処理100本ノック 公式サイト](https://nlp100.github.io/ja/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [spaCy](https://spacy.io/)
- [GiNZA](https://megagonlabs.github.io/ginza/)