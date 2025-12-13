import json
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import japanize_matplotlib

def run_regularization_experiment():
    """
    正則化パラメータCを変更しながら学習・評価を行い、グラフを描画する
    """
    # --- 1. データの準備 ---
    print("データを読み込んでいます...")
    data_dir = 'out/061'
    with open(os.path.join(data_dir, 'train_bow.json'), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, 'dev_bow.json'), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    # 特徴量とラベルの抽出
    X_train_dicts = [item['feature'] for item in train_data]
    y_train = [int(item['label']) for item in train_data]
    
    X_dev_dicts = [item['feature'] for item in dev_data]
    y_dev = [int(item['label']) for item in dev_data]

    # ベクトル化 (実験のため、ここで新規にFitします)
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train_dicts) #fitがあるものは新規
    X_dev = vectorizer.transform(X_dev_dicts) #fitがないから、既存のものを使っているという意味

    # --- 2. 実験設定 ---
    # Cの値の候補 (対数スケールで設定するのが一般的です)
    # Cが小さいほど正則化が強く(制約がきつく)、大きいほど弱く(自由に学習)なります
    c_candidates = [0.01, 0.1, 1, 10, 100]
    
    train_accuracies = []
    dev_accuracies = []

    print("実験を開始します (パラメータ数: {})".format(len(c_candidates)))

    # --- 3. 学習ループ ---
    for c in c_candidates:
        # モデルの定義と学習
        # solver='liblinear' は今回のデータ規模や正則化に適しています
        model = LogisticRegression(C=c, max_iter=1000, random_state=42, solver='liblinear')
        model.fit(X_train, y_train)
        
        # 予測とスコア算出
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_dev = accuracy_score(y_dev, model.predict(X_dev))
        
        train_accuracies.append(acc_train)
        dev_accuracies.append(acc_dev)
        
        print(f"  C = {c:<5} | Train Acc: {acc_train:.4f}, Dev Acc: {acc_dev:.4f}")

    # --- 4. 結果の保存とグラフ化 ---
    # 数値結果を保存
    result_file = 'out/result_69.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("C\tTrain_Acc\tDev_Acc\n")
        for c, t, d in zip(c_candidates, train_accuracies, dev_accuracies):
            f.write(f"{c}\t{t}\t{d}\n")
            
    # グラフ描画
    plt.figure(figsize=(8, 6))
    
    # 学習データ（青線）と検証データ（オレンジ線）をプロット
    plt.plot(c_candidates, train_accuracies, marker='o', label='Train Accuracy (学習)')
    plt.plot(c_candidates, dev_accuracies, marker='x', linestyle='--', label='Dev Accuracy (検証)')
    
    # 軸の設定
    plt.xscale('log')  # 横軸を対数スケールにする
    plt.xlabel('Regularization Parameter C (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Regularization Parameter C')
    plt.legend()
    plt.grid(True)
    
    # グラフ画像を保存
    graph_file = 'out/result_69.png'
    plt.savefig(graph_file)
    print(f"\n実験完了！\n数値結果: {result_file}\nグラフ画像: {graph_file}")

# --- 実行 ---
if __name__ == '__main__':
    run_regularization_experiment()