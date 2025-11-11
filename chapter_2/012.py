#メモリ効率重視
from collections import deque

n = 10

with open("popular-names.txt", "r", encoding="utf-8") as f:
    tail_lines = deque(f, maxlen=n)
for line in tail_lines:
    print(line.rstrip()) # 改行文字削除

# tail -n 10 popular-names.txt で行指定