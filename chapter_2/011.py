n = 10

with open("popular-names.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        print(line.rstrip())
        
# head -n 5 *.txt で、行指定