import json
import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data_and_predict(file_path, model, vectorizer):
    """
    データを読み込み、予測結果と正解ラベルのリストを返す関数
    """
    # データをロード
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 特徴量とラベルの抽出
    feature_dicts = [item['feature'] for item in data]
    y_true = [int(item['label']) for item in data]
    
    # ベクトル化と予測
    X = vectorizer.transform(feature_dicts)
    y_pred = model.predict(X)
    
    return y_true, y_pred

def calculate_scores(y_true, y_pred, data_name):
    """
    4つの指標を計算して辞書で返す関数
    """
    scores = {
        'Accuracy (正解率)': accuracy_score(y_true, y_pred),
        'Precision (適合率)': precision_score(y_true, y_pred),
        'Recall (再現率)': recall_score(y_true, y_pred),
        'F1 Score (F1)': f1_score(y_true, y_pred)
    }
    return scores

# --- メイン処理 ---
if __name__ == '__main__':
    # パス設定
    model_dir = 'out/062'
    path_model = os.path.join(model_dir, 'logistic_regression_model.joblib')
    path_vectorizer = os.path.join(model_dir, 'vectorizer.joblib')
    
    file_train = 'out/061/train_bow.json'
    file_dev = 'out/061/dev_bow.json'
    
    output_path = 'out/result_67.txt'
    
    # 1. モデルのロード
    print("モデルをロード中...")
    model = joblib.load(path_model)
    vectorizer = joblib.load(path_vectorizer)
    
    # 2. 学習データでの評価
    print("学習データ(train)を評価中...")
    y_true_train, y_pred_train = load_data_and_predict(file_train, model, vectorizer)
    scores_train = calculate_scores(y_true_train, y_pred_train, "学習データ")
    
    # 3. 検証データでの評価
    print("検証データ(dev)を評価中...")
    y_true_dev, y_pred_dev = load_data_and_predict(file_dev, model, vectorizer)
    scores_dev = calculate_scores(y_true_dev, y_pred_dev, "検証データ")
    
    # 4. 結果をDataFrameにまとめる
    df_scores = pd.DataFrame([scores_train, scores_dev], index=['学習データ (train)', '検証データ (dev)'])
    
    # 転置（行と列を入れ替え）して見やすくする
    df_scores = df_scores.T
    
    # 5. 表示と保存
    result_str = "--- 67. 評価指標の計測 ---\n\n" + df_scores.to_string() + "\n"
    
    print(result_str)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result_str)
        
    print(f"結果を保存しました: {output_path}")