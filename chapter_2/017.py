with open("popular-names.txt", "r") as f:
    lines = f.readlines()
    name = sorted(set(map(lambda x: x.split('\t')[0], lines))) #setで重複を削除、mapでリスト全体にラムダxを適用。
print(name)

#cut -f1 popular-names.txt | sort | uniq
#cut -f1：タブ区切りファイルの1列目だけを抽出
#sort:同じ名前が連続するように並べ替え
#uniq:重複を一つに
