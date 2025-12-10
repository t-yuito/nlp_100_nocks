import os
from gensim.models import KeyedVectors

model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

results = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)

os.makedirs('out', exist_ok=True)

output_path = 'out/result_log_53.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    for word, similarity in results:
        f.write(f"{word}\t{similarity:.4f}\n")

print("保存完了")