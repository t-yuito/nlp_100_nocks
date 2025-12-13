import os
import pandas as pd

def process_and_save_counts(file_path_train, file_path_dev, file_path_test, out_dir, out_file):
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, out_file)

    df_train = pd.read_csv(file_path_train, sep='\t')
    df_dev = pd.read_csv(file_path_dev, sep='\t')

    train_counts = df_train['label'].value_counts()
    dev_counts = df_dev['label'].value_counts()

    result_str = (
        "学習データの集計"
        f"{train_counts}\n\n"
        "検証データの集計"
        f"{dev_counts}\n\n"
    )

    print(result_str)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result_str)

    print("結果保存完了")

if __name__ == '__main__':
    file_path_train = '/home/takeuchi/workspace/nlp_100_nocks/chapter_7/SST-2/train.tsv'
    file_path_dev = '/home/takeuchi/workspace/nlp_100_nocks/chapter_7/SST-2/dev.tsv'
    file_path_test = '/home/takeuchi/workspace/nlp_100_nocks/chapter_7/SST-2/test.tsv'

    out_dir = 'out'
    out_file = 'result_60.txt'

    process_and_save_counts(file_path_train, file_path_dev, file_path_test, out_dir, out_file)