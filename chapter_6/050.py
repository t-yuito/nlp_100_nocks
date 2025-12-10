import os
from gensim.models import KeyedVectors

def main():
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
    output_dir = 'out'
    output_file = 'result_log_50.txt'
    target_word = 'United_States'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    if target_word in model:
        vector = model[target_word]

        print(f"{target_word}のベクトル")
        print(vector)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(vector))
        print("結果を保存しました。")
    else:
        print(f"エラー：{target_word}が含まれていません。")

if __name__ == '__main__':
    main()