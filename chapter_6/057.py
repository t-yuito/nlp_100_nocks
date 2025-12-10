import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def main():
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
    input_file = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/questions-words.txt'
    output_dir = 'out/057'
    output_txt = os.path.join(output_dir, 'result_log_57.txt')
    output_img = os.path.join(output_dir, 'cluster_visualization_57.png')

    os.makedirs(output_dir, exist_ok=True)

    print("モデルを読み込んでいます。")
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    except FileNotFoundError:
        print("モデルファイルが見つかりません。")
        return
    
    countries = set()
    target_sections = ['capital-common-countries', 'capital-world']
    is_target_section = False

    print("国名を抽出しています")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith(':'):
                    section_name = line.split()[1]
                    is_target_section = (section_name in target_sections)
                    continue
                
                if is_target_section:
                    parts = line.split()
                    if len(parts) >= 4:
                        countries.add(parts[1])
                        countries.add(parts[3])
    except FileNotFoundError:
        print(f"ファイルが見つかりません。")
        return

    country_list = []
    country_vectors = []
    for country in countries:
        country_list.append(country)
        country_vectors.append(model[country])

    print(f"対象国名数： {len(country_list)}")

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(country_vectors)
    labels = kmeans.labels_ #fit後に決定した値には_をつける。

    df_results = pd.DataFrame({
        'Country': country_list,
        'Cluster': labels
    })

    with open(output_txt, 'w') as f:
        f.write("result")
        for i in range(5):
            cluster_countries = df_results[df_results['Cluster'] == i]['Country'].values #.valuesは、抜き出したものをリストにする。
            line = f"\n[Cluster {i}]\n" + ", ".join(cluster_countries) + "\n"
            f.write(line)
            print(f"[Cluster {i}] {', '.join(cluster_countries[:5])} ...")

    print("画像を作成")

    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(country_vectors)

    plt.figure(figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i in range(len(country_list)):
        cluster_idx = labels[i]
        x = vectors_2d[i, 0]
        y = vectors_2d[i, 1]

        plt.scatter(x, y, c=colors[cluster_idx], alpha=0.6)

    plt.title('Country_Vectors_Clustering (k=5)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_img)
    plt.close() #メモリ解放

if __name__ == '__main__':
    main()