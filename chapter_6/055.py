import os
from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm

MODEL_PATH = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
DATA_PATH = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/questions-words.txt'
OUTPUT_DIR = 'out/055'

def load_data_with_category(filename):
    data = []
    current_category = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() #先頭、末尾にある空白文字を削除する。なので、文章中にある空行は消えない。

            if line.startswith(':'):
                if line.startswith(': gram'):
                    current_category = 'syntactic'
                else:
                    current_category = 'semantic'
                continue

            words = line.split()
            if len(words) == 4:
                data.append({
                    "v1": words[0],
                    "v2": words[1],
                    "v3": words[2],
                    "v4": words[3],
                    "category": current_category
                })
    return data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"モデル読み込み開始")
    global model #関数の中だけでなく、全体で共有できるようにする。
    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

    print("データ読み込み開始")
    data = load_data_with_category(DATA_PATH)
    df = pd.DataFrame(data)

    print(f"データ件数：{len(df)}件")

    def get_most_similar(row): #これをmain関数の外に出した場合、global mainとしてないとエラーになる。
        try:
            res = model.most_similar(positive=[row['v2'], row['v3']], negative=[row['v1']], topn=1)[0] #model.most_similarは、リストの中にタプルの形で出力される。
            return res[0], res[1] #単語、類似度
        except KeyError:
            return None, None
    
    tqdm.pandas(desc='推論実行中') #pandasにtqdm機能を追加。
    df[['pred_word', 'score']] = df.progress_apply(get_most_similar, axis=1, result_type='expand') 
    #progressは、tqdmを使用するためにある。applyは1行または1列ずつ実行していく際に必要。expandは、単語、類似度というタプルが一つずつカラムに入るようにする。
    
    #計算できなかった行を削除
    original_len = len(df)
    df.dropna(subset=['pred_word'], inplace=True)
    print(f"計算完了：{len(df)}/{original_len}件 語彙不足でスキップ：{original_len - len(df)}")

    df['is_correct'] = df['pred_word'] == df['v4']

    print("\n---正解率---")

    total_acc = df['is_correct'].mean()
    print(f"全体正解率：{total_acc:.4f}")

    category_acc = df.groupby('category')['is_correct'].mean()
    print(f"意味的アナロジー：{category_acc['semantic']:.4f}")
    print(f"文法的アナロジー：{category_acc['syntactic']:.4f}")

    detail_path = os.path.join(OUTPUT_DIR, 'result_log_55_details.csv')
    df.to_csv(detail_path, index=False)

    score_path = os.path.join(OUTPUT_DIR, 'result_log_55.txt')
    with open(score_path, 'w') as f:
        f.write(f"Total： {total_acc:.4f}\n")
        f.write(f"Semantic： {category_acc['semantic']:.4f}\n")
        f.write(f"Syntactic： {category_acc['syntactic']:.4f}\n")

    print("詳細結果を保存完了")

if __name__ == '__main__':
    main()