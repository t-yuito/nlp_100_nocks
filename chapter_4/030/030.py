import spacy

# モデルの読み込み
nlp = spacy.load('ja_ginza')

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

doc = nlp(text)

# 動詞の基本形（辞書形）のみをリストに抽出
# token.text ではなく token.lemma_ を使うことで「した」→「する」のように正規化されます
verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

# ファイルに書き込み（動詞のみ）
output_file = 'verbs_only.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    # リストの中身を改行で結合して書き込む
    f.write('\n'.join(verbs))

print(f"抽出完了: {output_file} に動詞のみを保存しました。")
print(f"抽出された動詞: {verbs}")