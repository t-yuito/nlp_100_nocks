import os
from gensim.models import KeyedVectors

def main():
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
    data_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/questions-words.txt'
    output_dir = 'out'
    output_file = 'result_log_54.txt'
    target_section = ': capital-common-countries'

    # download_data(data_file)  # 必要に応じて関数を定義するか削除

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    results = []
    is_target_section = False

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                if line == target_section:  
                    is_target_section = True
                    print(f"セクション{target_section}の処理を開始します。")
                else:
                    is_target_section = False
                continue
            
            if not is_target_section:
                continue
            
            words = line.split()
            if len(words) != 4:
                continue
            
            word1, word2, word3, word4 = words

            try:
                prediction = model.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0]
                pred_word = prediction[0]
                similarity = prediction[1]

                result_line = f"{line} {pred_word} {similarity:.4f}"
                results.append(result_line)
            
            except KeyError as e:
                print(f"モデルに単語がないときはスキップ: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"保存完了：{len(results)}件")

if __name__ == '__main__':
    main()