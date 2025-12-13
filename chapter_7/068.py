import joblib
import os
import pandas as pd

def show_top_features(model_path, vectorizer_path, output_file):
    """
    重みの高い単語と低い単語のトップ20を表示・保存する
    """
    # 1. モデルとVectorizerのロード
    print("モデルをロード中...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 2. 特徴量名（単語）と重みの抽出
    # get_feature_names_out() で単語のリストを取得できます
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]  # ロジスティック回帰の重み係数
    
    # 3. データフレームにまとめてソート
    df = pd.DataFrame({
        'word': feature_names,
        'weight': weights
    })
    
    # 重みで降順ソート（高い順）
    df_sorted = df.sort_values('weight', ascending=False) #あるカラムを基準にソートする。
    
    # 4. 上位と下位の抽出
    top_20_positive = df_sorted.head(20)  # 重みが最も高い（ポジティブ寄り）
    top_20_negative = df_sorted.tail(20)  # 重みが最も低い（ネガティブ寄り）
    
    # --- 結果の文字列作成 ---
    result_str = "--- 68. 特徴量の重みの確認 ---\n\n"
    
    # ポジティブ（重みが高い）
    result_str += "【重みが高い特徴量 (ポジティブ寄り) Top 20】\n"
    result_str += top_20_positive.to_string(index=False)
    result_str += "\n\n" + "="*40 + "\n\n"
    
    # ネガティブ（重みが低い）
    # 見やすいように、下位20件は「逆順（絶対値が大きい順）」に並べ直して表示します
    result_str += "【重みが低い特徴量 (ネガティブ寄り) Top 20】\n"
    result_str += top_20_negative.iloc[::-1].to_string(index=False)
    result_str += "\n"
    
    # 5. 表示と保存
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
    
    output_path = 'out/result_68.txt'
    
    # 実行
    show_top_features(path_model, path_vectorizer, output_path)