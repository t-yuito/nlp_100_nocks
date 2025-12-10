#高次元データの可視化には、t-SNEを使う。
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import os
from adjustText import adjust_text

def main():
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_6/GoogleNews-vectors-negative300.bin'
    input_file = 'questions-words.txt'
    output_dir = 'out'
    output_img = os.path.join(output_dir, 'cluster_tsne_59.png')

    os.makedirs(output_dir, exist_ok=True)

    print("モデルの読み込み")

    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    except FileNotFoundError:
        print("モデルファイルが見つかりません。")
        return
    
    countries = set()
    target_sections = ['capital-common-countries', 'capital-world']
    is_target_section = False

    print("国名を抽出しています。")
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
        print(f"{input_file}が見つかりません。")

    country_list = []
    country_vectors = []
    for country in countries:
        if country in model:
            country_list.append(country)
            country_vectors.append(model[country])

    print(f"対象国名数： {len(country_list)}")

    print("t-SNEを実行")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    vectors_2d = tsne.fit_transform(np.array(country_vectors)) #scikit-learnはnumpy形式を好む。

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(country_vectors)

    print("画像を生成中")
    plt.figure(figsize=(20, 20))

    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='jet', alpha=0.7) #tsneで、x,y用にすでに変換されている。
    
    texts = []
    for i, country in enumerate(country_list):
        texts.append(plt.text(vectors_2d[i, 0], vectors_2d[i, 1], country, fontsize=9))

    print("文字の位置を自動調整")
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title('t-SNE Visualization of Country Vectors')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_img)
    plt.close()
    
    print("完了しました。")

if __name__ == "__main__":
    main()