import zipfile
import os
import glob
from gensim.models import KeyedVectors

OUT_DIR = "out"
LOG_FILE = os.path.join(OUT_DIR, "result_log_50.txt")

def log_print(message, file_obj):
    """
    コンソールとファイルの両方にメッセージを出力する関数
    """
    print(message)
    file_obj.write(message + "\n")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(LOG_FILE, "w", encoding='utf-8') as f:
        log_print("処理を開始します。", f)

        bin_files = glob.glob("*.bin")

        if not bin_files:
            print("エラー：binファイルが見つかりません。")
            return

        model_path = bin_files[0]
        print("binファイル読み込み中")

        try:
            model = KeyedVectors.load_word2vec_format(model_path, binary=True)

            word1 = "United_States"
            word2 = "U.S."

            if word1 in model and word2 in model:
                similarity = model.similarity(word1, word2)

                log_print("-"*30, f)
                log_print(f"Target: {word1}, {word2}", f)
                log_print(f"Cosine Similarity: {similarity:.4f}", f)
            else:
                log_print("エラー：指定単語が辞書にありません。", f)
        
        except Exception as e:
            log_print(f"予期せぬエラー： {e}", f)

if __name__ == "__main__":
    main()
            
