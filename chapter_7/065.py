import joblib
import os
from collections import Counter

def predict_sentiment(text, model_path, vectorizer_path, output_file):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    words = text.split(' ')
    feature_dict = dict(Counter(words))

    X = vectorizer.transform([feature_dict]) #辞書をリストで囲んで渡す。

    pred_label = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    sentiment = "ポジティブ (1)" if pred_label == 1 else "ネガティブ (0)"

    result_str = (
        "--- 65. 任意のテキストの予測 ---\n"
        f"入力テキスト: {text}\n\n"
        "【予測結果】\n"
        f"  判定: {sentiment}\n"
        f"  確率: ネガティブ={pred_proba[0]:.4f}, ポジティブ={pred_proba[1]:.4f}\n"
    )

    print(result_str)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_str)
    print(f"結果を保存しました: {output_file}")

if __name__ == '__main__':
    model_dir = 'out/062'
    path_model = os.path.join(model_dir, 'logistic_regression_model.joblib')
    path_vectorizer = os.path.join(model_dir, 'vectorizer.joblib')

    output_path = 'out/result_65.txt'
    
    target_text = "the worst movie I 've ever seen"
    
    predict_sentiment(target_text, path_model, path_vectorizer, output_path)