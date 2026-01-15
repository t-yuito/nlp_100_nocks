import pandas as pd
import torch
import importlib

import sys
sys.path.insert(0, '/home/takeuchi/workspace/nlp_100_nocks/chapter_8')
module = importlib.import_module('070')
load_word_embeddings = module.load_word_embeddings

def load_and_process_sst(file_path, word_to_id):
    df = pd.read_csv(file_path, sep='\t')

    dataset = []

    # df.iterrows()で行ごとに反復処理する。
    for _, row in df.iterrows():
        text = row['sentence']
        label = row['label']

        # テキストをトークン化し、IDに変換
        input_ids = [word_to_id[word] for word in text.split() if word in word_to_id]

        if len(input_ids) == 0:
            continue

        data_item = {
            'text': text,
            'label': torch.tensor([float(label)]),
            'input_ids': torch.tensor(input_ids, dtype=torch.long)
        }
        dataset.append(data_item)
    
    return dataset

if __name__ == '__main__':
    import os
    
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/GoogleNews-vectors-negative300.bin.gz'
    E, w2id, id2w = load_word_embeddings(model_path, limit=300000)

    train_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/train.tsv'
    dev_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/SST-2/dev.tsv'

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_071.txt')

    try:
        train_dataset = load_and_process_sst(train_file, w2id)
        dev_dataset = load_and_process_sst(dev_file, w2id)

        results = [
            f"訓練データ数: {len(train_dataset)}",
            f"開発データ数: {len(dev_dataset)}",
            "",
            "--- データ例 ---",
            str(train_dataset[0])
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in results:
                print(line)
                f.write(line + '\n')
        
        print(f"\n結果を保存しました: {output_file}")

    except FileNotFoundError:
        error_msg = "エラー: train.tsv または dev.tsv が見つかりません。\nGLUEデータをダウンロードして配置してください。"
        print(error_msg)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)