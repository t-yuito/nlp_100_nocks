import math
import os

N = int(input())
filename = "popular-names.txt"
output_dir = "splits_code"

os.makedirs(output_dir, exist_ok=True)

with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

chunk_size = math.ceil(len(lines) / N) # 切り上げ

for i in range(N):
    start = i * chunk_size
    end = start + chunk_size
    chunk = lines[start:end]

    output_path = os.path.join(output_dir, f"split_{i+1:02d}.txt")
    with open(output_path, "w", encoding="utf-8") as out: # "w"は書き込み
        out.writelines(chunk) #writelines:各要素を連結して書くメソッド

#unixコマンド split -n r/10 popular-names.txt splits_unix/output_
#r:round-robinの略で、各ファイルに順番に一つずつ要素が格納されていくため、均等となる。 -lだと、均等にならなかった。