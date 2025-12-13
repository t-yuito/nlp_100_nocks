import json
import joblib
import os

def predict_first_instance(model_path, vectorizer_path, data_path, output_file):
    print("モデル読み込み中")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    first_instance = dev_data[0]

    feature_dict = first_instance['feature']
    X_val = vectorizer.transform([feature_dict]) #fitしたものと同じvectorizerを使う。

    pred_label = model.predict(X_val)[0]

    pred_proba = model.predict_proba(X_val)[0]

    true_label = int(first_instance['label'])
    text = first_instance['text']

    result_str = (
        "--- 予測結果 (Task 63) ---\n"
        f"テキスト: {text}\n"
        f"確率(0/1): {pred_proba}\n"
        f"予測ラベル: {pred_label} ({'ポジティブ' if pred_label==1 else 'ネガティブ'})\n"
        f"正解ラベル: {true_label}\n"
    )

    if pred_label == true_label:
        result_str += ">> 判定: 正解！ (一致しています)\n"
    else:
        result_str += ">> 判定: 不正解... (一致していません)\n"

    print(result_str)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_str)
    print("結果を保存しました。")

if __name__ == '__main__':
    file_dev_json = 'out/061/dev_bow.json'

    dir_model = 'out/062'
    path_model = os.path.join(dir_model, 'logistic_regression_model.joblib')
    path_vectorizer = os.path.join(dir_model, 'vectorizer.joblib')

    output_path = 'out/result_63.txt'

    predict_first_instance(path_model, path_vectorizer, file_dev_json, output_path)