import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

def main():
    dataset_path = 'wordsim353/combined.csv'
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'

    output_dir = 'out'
    output_file = os.path.join(output_dir, 'result_log_56.txt')
    os.makedirs(output_dir, exist_ok=True)

    print("データを読み込んでいます。")
    df = pd.read_csv(dataset_path)
    
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    except FileNotFoundError:
        err_msg = f"エラー：ファイルが見つかりません。パスを確認してください。"
        print(err_msg)
        with open(output_file, 'w') as f:
            f.write(err_msg)
        return

    def calculate_vector_similarity(row):
        w1 = row['Word 1']
        w2 = row['Word 2']
        if w1 in model and w2 in model:
            return model.similarity(w1, w2)
        else:
            return np.nan
    
    print("類似度を計算中")
    df['vector_sim'] = df.apply(calculate_vector_similarity, axis=1)

    df_clean = df.dropna(subset=['vector_sim'])

    correlation, pvalue = spearmanr(df_clean['Human (mean)'], df_clean['vector_sim'])

    result_str = (
        f"result\n"
        f"Number of pairs calculated: {len(df_clean)} (Total: {len(df)})\n"
        f"Spearman Correlation: {correlation:.4f}\n"
        f"P-value: {pvalue:.4e}\n"   
    )

    print(result_str)

    try:
        with open(output_file, 'w') as f:
            f.write(result_str)
        print("結果を保存しました。")
    except Exception as e:
        print(f"ファイル保存エラー {e}")

if __name__ == "__main__":
    main()