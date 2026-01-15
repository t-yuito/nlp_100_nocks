# Chapter 6: 単語ベクトル

## 概要
学習済み単語ベクトル（Google News Word2Vec）を使った単語の意味解析問題です。

## 問題一覧

| 問題番号 | タイトル | 内容 |
|----------|----------|------|
| 50 | 単語ベクトルの読み込み | gensimで単語ベクトルを読み込み、"United_States"のベクトルを表示 |
| 51 | 単語の類似度 | "United_States"と"U.S."のコサイン類似度を計算 |
| 52 | 類似単語上位10件 | "United_States"に類似した単語Top10を取得 |
| 53 | 加法構成性 | Spain - Madrid + Athens = Greece（アナロジー計算）|
| 54 | アナロジーデータの読み込み | questions-words.txtから首都-国のペアを抽出 |
| 55 | アナロジータスクの評価 | 意味的・文法的アナロジーの正解率を計算 |
| 56 | WordSimilarity-353 | 人間の類似度評価との相関（Spearman相関）|
| 57 | k-meansクラスタリング | 国名ベクトルをk-meansで5クラスタに分類 |
| 58 | Ward法クラスタリング | 階層的クラスタリングとデンドログラムの描画 |
| 59 | t-SNEによる可視化 | 高次元ベクトルを2次元に圧縮して可視化 |

## 学習ポイント
- gensimによる単語ベクトルの読み込み
- コサイン類似度の計算
- 単語アナロジー（King - Man + Woman = Queen）
- scikit-learnによるクラスタリング（k-means, Ward法）
- 次元削減（PCA, t-SNE）
- Spearman順位相関係数
