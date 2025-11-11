n = 10

with open("popular-names.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f): #先頭10行の時のみ可能
        if i >= n:
            break
        print(line.replace("\t", " "), end="") #end=""で、printの自動改行制御

#head popular-names.txt | tr '\t' ' '
#head -n 10 popular-names.txt | expand -t 1