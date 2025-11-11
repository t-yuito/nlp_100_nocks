with open("popular-names.txt", "r") as f:
    lines = f.readlines() # リスト変換
    count = len(lines) 
    print(count)

# wc *.txt で、-l:lines, -w:word, -m:文字数 がわかる。