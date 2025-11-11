with open("popular-names.txt", "r") as f:
    lines = [line.strip() for line in f]


sorted_lines = sorted(lines, key=lambda x: int(x.split('\t')[2]), reverse=True)

with open("sorted_by_col3_code.txt", "w") as out:
    out.write("\n".join(sorted_lines))

#.join:リストの要素を特定の文字で結合。"\n".join：各行の間に\nを 挟んで一本の長い文字列へ


# sort -k3,3nr popular-names.txt > sorted_by_col3_unix.txt
# -k3,3:ソートキーに3列目だけを指定