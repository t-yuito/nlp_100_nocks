import json
import joblib
import os
import numpy as np

def show_conditional_probability(model_path, vectorizer_path, data_path, output_file):
    """
    検証データの先頭事例について、各ラベルの条件付き確率を表示・保存する
    """
    # 1. モデルとVectorizerのロード
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 2. データのロード
    with open(data_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # 先頭の事例
    instance = dev_data[0]
    feature_dict = instance['feature']
    text = instance['text']
    
    # 3. ベクトル化
    X = vectorizer.transform([feature_dict])
    
    # 4. 条件付き確率の計算 (predict_proba)
    # 戻り値は [[P(Y=0), P(Y=1)]] の形になっているから、[]を一つのけたい。
    probs = model.predict_proba(X)[0]
    
    prob_neg = probs[0]  # ラベル0 (ネガティブ) の確率
    prob_pos = probs[1]  # ラベル1 (ポジティブ) の確率
    
    # 5. 結果の作成
    # 読みやすいように%表記もつけます
    result_str = (
        "--- 64. 条件付き確率の算出 ---\n"
        f"テキスト: {text}\n\n"
        "【算出結果】\n"
        f"  P(Y=0 | X) [ネガティブ]: {prob_neg:.6f} ({prob_neg*100:.2f}%)\n"
        f"  P(Y=1 | X) [ポジティブ]: {prob_pos:.6f} ({prob_pos*100:.2f}%)\n\n"
        "【判定】\n"
        f"  予測ラベル: {'1 (ポジティブ)' if prob_pos > prob_neg else '0 (ネガティブ)'}\n"
    )
    
    # 6. 表示と保存
    print(result_str)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_str)
        
    print(f"結果を保存しました: {output_file}")

# --- メイン処理 ---
if __name__ == '__main__':
    # ファイルパス設定
    input_file = 'out/061/dev_bow.json'
    model_dir = 'out/062'
    path_model = os.path.join(model_dir, 'logistic_regression_model.joblib')
    path_vectorizer = os.path.join(model_dir, 'vectorizer.joblib')
    
    # 出力先
    output_path = 'out/result_64.txt'

    # 実行
    show_conditional_probability(path_model, path_vectorizer, input_file, output_path)