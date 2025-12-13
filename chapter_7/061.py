import pandas as pd
from collections import Counter
import pprint #辞書を見やすく表示するライブラリ
import os
import json

def text_to_bow_dataset(file_path):
    dataset = []

    df = pd.read_csv(file_path, sep='\t')

    for _, row in df.iterrows(): #_にはindex, rowには、dfが1行ずつ上から格納されていく。
        text = row['sentence']

        label = str(row['label']) #カウンターを使うため、文字列にする

        words = text.split(' ') #スペース区切りで単語セット作成

        feature = dict(Counter(words)) #辞書にして、counterオブジェクトを変換

        instance = {
            'text': text,
            'label': label,
            'feature': feature
        }

        dataset.append(instance)

    return dataset

def save_list_to_json(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False とすると、日本語がそのまま読める形で保存できる。
        # indent=2で、人間が見やすい形式で改行・インデントされる。
        json.dump(data_list, f, ensure_ascii=False, indent=2)

if __name__ == '__main__': #このファイルをimport した時に、勝手に関数が実行されないようにするため。
    file_path_train = '/home/takeuchi/workspace/nlp_100_nocks/chapter_7/SST-2/train.tsv'
    file_path_dev = '/home/takeuchi/workspace/nlp_100_nocks/chapter_7/SST-2/dev.tsv'

    output_dir = 'out/061'
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = text_to_bow_dataset(file_path_train)
    dev_dataset = text_to_bow_dataset(file_path_dev)

    save_list_to_json(train_dataset, os.path.join(output_dir, 'train_bow.json'))
    save_list_to_json(dev_dataset, os.path.join(output_dir, 'dev_bow.json'))

    print("保存完了")