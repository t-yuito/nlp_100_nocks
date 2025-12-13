import json
import os
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def train_model(train_data):
    feature_dicts = [item['feature'] for item in train_data]
    labels = [int(item['label']) for item in train_data]

    vectorizer = DictVectorizer()

    print("特徴ベクトルを変換中")
    X_train = vectorizer.fit_transform(feature_dicts)

    print("モデルを学習中")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, labels)

    print(f"学習完了 データ数{X_train.shape[0]}、特徴量数{X_train.shape[1]}")

    return model, vectorizer

if __name__ == '__main__':
    input_file = 'out/061/train_bow.json'
    output_dir = 'out/062'
    model_file = 'logistic_regression_model.joblib'
    vectorizer_file = 'vectorizer.joblib'

    print("データを読み込んでいます。")
    train_data = load_dataset(input_file)

    model, vectorizer = train_model(train_data)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, model_file))
    joblib.dump(vectorizer, os.path.join(output_dir, vectorizer_file))

    print(f"モデルと変換器を{output_dir}に保存しました。")