from collections import Counter

with open("popular-names.txt", "r") as f:
    names = [line.split('\t')[0] for line in f]

counter = Counter(names) #

for name, count in counter.most_common(): #出現回数の多い順にソート
    print(f"{name}\t{count}")

# Counter：要素→出現回数の辞書形式に変換してくれる関数。

# cut -f1 popular-names.txt | sort | uniq -c | sort -nr | head

# uniq -c：重複をカウント
# sort -nr：出現回数(数字)を降順に並べ替え r:reverse