import gzip
import json
import re
import spacy
import csv
import math
from collections import Counter

nlp = spacy.load('ja_ginza', disable=['ner', 'parser'])

def remove_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[http.+?\]", "", text)
    text = re.sub(r"\{\{.+?\}\}", "", text)
    text = re.sub(r"={2,}", "", text)
    text = re.sub(r"<.+?>", "", text)
    return text

input_file = '../jawiki-country.json.gz'
output_file = 'rfidf_top20.csv'

df_counter = Counter()
japan_word_count = Counter()
total_docs = 0

with gzip.open(input_file, 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            title = data.get('title', '')
            text = remove_markup(data.get('text', ''))

            current_doc_nouns = set()

            is_japan = (title == '日本')

            for sentence in text.split('\n'):
                if not sentence.strip():
                    continue

                chunks = [sentence]
                for chunk in chunks:
                    doc = nlp(chunk)
                    stop_words = ['|', '}', '{', '%', 'file', 'File', '*', '(', ')']
                    noun_tokens = [
                        token.lemma_ for token in doc
                        if token.pos_ in ['NOUN', 'PROPN'] and token.text not in stop_words
                    ]

                    current_doc_nouns.update(noun_tokens)

                    if is_japan:
                        japan_word_count.update(noun_tokens)

            df_counter.update(current_doc_nouns)
            total_docs += 1
        
        except json.JSONDecodeError:
            continue

tfidf_scores = []


total_terms_in_japan = sum(japan_word_count.values())

for word, count in japan_word_count.items():
    tf = count / total_terms_in_japan

    df = df_counter[word]
    idf = math.log10(total_docs / (df + 1)) + 1

    tfidf = tf * idf

    tfidf_scores.append({
        'word': word,
        'tf': tf,
        'idf': idf,
        'tfidf': tfidf,
        'count': count,
        'df': df
    })

tfidf_scores.sort(key=lambda x: x['tfidf'], reverse=True)

with open(output_file, "w", newline="", encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['順位', '単語', 'TF-IDF', 'TF', 'IDF', '出現数(日本)', '出現記事数(DF)'])

    for rank, item in enumerate(tfidf_scores[:20], 1):
        print(f"{rank:<4} | {item['word']:<10} | {item['tfidf']:.5f} | {item['tf']:.5f} | {item['idf']:.5f}")

        writer.writerow([
            rank,
            item['word'],
            f"{item['tfidf']:.5f}",
            f"{item['tf']:.5f}",
            f"{item['idf']:.5f}",
            item['count'],
            item['df']
        ])

print('終了')