import numpy as np
from gensim.models import KeyedVectors

def load_word_embeddings(model_path, limit=None):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=limit)

    vocab = model.index_to_key
    dim = model.vector_size
    v = len(vocab)

    # 0行目を0にする。
    embedding_matrix = np.zeros((v + 1, dim), dtype=np.float32)
    
    # IDと単語の双方向マッピング
    word_to_id = {"<PAD>": 0}
    id_to_word = {0: "<PAD>"}

    for i, word in enumerate(vocab, start=1):
        embedding_matrix[i] = model[word]
        word_to_id[word] = i
        id_to_word[i] = word

    return embedding_matrix, word_to_id, id_to_word


if __name__ == "__main__":
    import os
    
    model_path = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/GoogleNews-vectors-negative300.bin.gz'
    E, w2id, id2w = load_word_embeddings(model_path, limit=100000)

    output_dir = '/home/takeuchi/workspace/nlp_100_nocks/chapter_8/out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result_070.txt')

    results = [
        f"行列の形状: {E.shape}",
        f"ID 0 のベクトル (PAD): {E[0][:5]}...",
        f"ID 1 の単語: {id2w[1]}"
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            print(line)
            f.write(line + '\n')
    
    print(f"\n結果を保存しました: {output_file}")