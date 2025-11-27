import json
import gzip
import re
import spacy
from collections import Counter
import matplotlib.pyplot as plt

nlp = spacy.load('ja_ginza', disable=['ner', 'parser'])

def remove_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[http.+?\]", "", text)
    text = re.sub(r"\{\{.+?\}\}", "", text)
    text = re.sub(r"={2,}", "", text)
    text = re.sub(r"<.+?>", "", text)
    return text

def read_wiki_corpus(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                raw_text = data.get('text', '')
                clean_text = remove_markup(raw_text)

                for sentence in clean_text.split('\n'):
                    if not sentence.strip():
                        continue
                    yield sentence
            
            except json.JSONDecodeError:
                continue

input_file = '../jawiki-country.json.gz'
word_freq = Counter()

for doc in nlp.pipe(read_wiki_corpus(input_file), batch_size=50):
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ not in ['PUNCT', 'SYM', 'SPACE']
    ]
    word_freq.update(tokens)

counts = [count for word, count in word_freq.most_common()]

ranks = range(1, len(counts) + 1)

plt.figure(figsize=(10, 6))
plt.scatter(ranks, counts, s=5, alpha=0.5)

plt.xscale('log')
plt.yscale('log')

plt.title("Zipf's Law (Wikipedia Corpus)")
plt.xlabel('Rank(log)')
plt.ylabel('Frequency(log)')
plt.grid(which="both", linestyle="--", alpha=0.5)

output_img = 'zipf_law.png'
plt.savefig(output_img)
print("保存完了")