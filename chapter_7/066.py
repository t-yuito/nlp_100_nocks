import json
import joblib
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(model_path, vectorizer_path, data_path, output_file):
    """
    検証データの混同行列を作成し、保存する関数
    """
    # 1. モデルとVectorizerのロード
    print(f"モデルをロード中...: {model_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 2. 検証データの読み込み
    with open(data_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # 3. データ整形 (リスト形式で一括処理)
    # dev_data全体から特徴量と正解ラベルを取り出す
    feature_dicts = [item['feature'] for item in dev_data]
    y_true = [int(item['label']) for item in dev_data]
    
    # 4. ベクトル化 (transform)
    # ※ここでは一括変換するので、前回のような [0] は不要です
    X_val = vectorizer.transform(feature_dicts)
    
    # 5. 予測 (一括予測)
    y_pred = model.predict(X_val)
    
    # 6. 混同行列の作成
    # labels=[0, 1] で順序を固定 (0=ネガティブ, 1=ポジティブ)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # 見やすくするためにDataFrameにする
    cm_df = pd.DataFrame(
        cm, 
        index=['正解:0(ネガティブ)', '正解:1(ポジティブ)'], 
        columns=['予測:0(ネガティブ)', '予測:1(ポジティブ)']
    )
    
    # 7. 結果の文字列作成
    result_str = (
        "--- 66. 混同行列 (Confusion Matrix) ---\n\n"
        f"{cm_df}\n\n"
        "【解説】\n"
        f"・TN (True Negative): {cm[0, 0]} 件 (ネガティブを正しくネガティブと予測)\n"
        f"・FP (False Positive): {cm[0, 1]} 件 (ネガティブなのにポジティブと誤認)\n"
        f"・FN (False Negative): {cm[1, 0]} 件 (ポジティブなのにネガティブと誤認)\n"
        f"・TP (True Positive): {cm[1, 1]} 件 (ポジティブを正しくポジティブと予測)\n"
    )
    
    # 8. 表示と保存
    print(result_str)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_str)
        
    print(f"結果を保存しました: {output_file}")

# --- メイン処理 ---
if __name__ == '__main__':
    # パス設定
    model_dir = 'out/062'
    path_model = os.path.join(model_dir, 'logistic_regression_model.joblib')
    path_vectorizer = os.path.join(model_dir, 'vectorizer.joblib')
    file_dev_json = 'out/061/dev_bow.json'
    
    # 出力先
    output_path = 'out/result_66.txt'
    
    # 実行
    create_confusion_matrix(path_model, path_vectorizer, file_dev_json, output_path)