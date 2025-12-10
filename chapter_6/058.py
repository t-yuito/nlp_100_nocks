import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from gensim.models import KeyedVectors
import os

def main():
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
    input_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/questions-words.txt'
    output_dir = 'out'
    output_img = os.path.join(output_dir, 'cluster_dendrogram_58.png')

    os.makedirs(output_dir, exist_ok=True)

    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    except FileNotFoundError:
        print("モデルファイルが見つかりません。")
        return

    countries = set()
    target_sections = ['capital-common-countries', 'capital-world']
    is_target_section = False

    print("国名探し中")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith(':'):
                    section_name = line.split()[1]
                    is_target_section = (section_name in target_sections)
                    continue #ここでcontinueを入れないと、題名に対しても国名抽出処理をしようとしてしまう。
                
                if is_target_section:
                    parts = line.split()
                    if len(parts) >= 4:
                        countries.add(parts[1])
                        countries.add(parts[3])
    except FileNotFoundError:
        print(f"{input_file}が見つかりません。")
        return

    country_list = []
    country_vectors = []
    for country in countries:
        if country in model:
            country_list.append(country)
            country_vectors.append(model[country])
    print(f"対象国名数: {len(country_list)}")

    print("Ward法クラスタリング実行")
    Z = linkage(country_vectors, method='ward')

    print("画像を生成中")

    plt.figure(figsize=(25, 10))

    dendrogram(
        Z,
        labels=country_list,
        leaf_rotation=90,
        leaf_font_size=10
    )

    plt.title('Hierarchical Clustering Dendrogram (Ward Method)')
    plt.xlabel('Country')
    plt.ylabel('Distance')
    plt.tight_layout()

    plt.savefig(output_img)
    plt.close()

if __name__ == '__main__':
    main()