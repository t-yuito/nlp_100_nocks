import random

with open("popular-names.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

with open("randomized.txt", "w") as out:
    out.writelines(lines)

# shuf popular-names.txt | head -5