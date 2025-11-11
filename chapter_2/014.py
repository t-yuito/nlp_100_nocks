n = 10

with open("popular-names.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        cols = line.strip().split("\t") #タブで区切る
        print(cols[0])

#head popular-names.txt | cut -f 1 1列目(1フィールド)だけを取り出す