import os
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai
from datasets import load_dataset
from tqdm import tqdm
import time
import random

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
TASK_NAME = "machine_learning"
FILE_NAME = "answer_history_043.txt" # ファイル名変更
LIMIT_NUM = 10  # 必要に応じて増やしてください

# ---------------------------------------------------------
# 準備
# ---------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")

dataset_url = f"https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/{TASK_NAME}.csv"

print("データセット読み込み")
try:
    dataset = load_dataset(
        "csv", 
        data_files=dataset_url, 
        split="train", 
        column_names=["question", "A", "B", "C", "D", "answer"]
    )
except Exception as e:
    print(f"エラーが発生しました：{e}")
    exit()
    
if LIMIT_NUM is not None:
    dataset = dataset.select(range(LIMIT_NUM))

# ---------------------------------------------------------
# 実験用プロンプト作成関数 (正解をDに固定する版)
# ---------------------------------------------------------
def create_biased_prompt(record):
    question = record['question']
    
    # 1. 元の正解記号と選択肢を取得
    original_answer_char = record['answer'].strip().upper() # 例: "A"
    original_options = {
        "A": record['A'],
        "B": record['B'],
        "C": record['C'],
        "D": record['D']
    }
    
    # 2. 正解の選択肢の「文章」を取得
    correct_content = original_options[original_answer_char]
    
    # 3. 不正解（ハズレ）の選択肢の「文章」をリスト化
    distractors = []
    for char, content in original_options.items():
        if char != original_answer_char:
            distractors.append(content)
            
    # 4. 新しい選択肢リストを作成
    # 【実験設定】 ハズレ3つを先に並べ、最後に正解(correct_content)を置く
    # これにより、正解は必ず4番目になる
    new_options_list = distractors + [correct_content]
    
    # 5. プロンプト用の文字列を作成
    option_str = ""
    chars = ["A", "B", "C", "D"] # 新しい記号の割り当て
    
    # new_options_listの 0,1,2番目はハズレ、3番目(D)が正解
    for i, opt in enumerate(new_options_list):
        option_str += f"{chars[i]}. {opt}\n"

    prompt = f"""以下は{TASK_NAME}に関する多肢選択問題です。
正解と思われる選択肢の記号（A, B, C, D）のみを答えてください。
解説や余計な文言は一切不要です。

問題: {question}

選択肢:
{option_str}

答え:"""
    
    # 新しい正解は常に "D" (リストの最後に追加したため)
    return prompt, "D"

# ---------------------------------------------------------
# 推論ループ
# ---------------------------------------------------------
correct_count = 0
total_count = len(dataset)
results_log = []

print(f"全 {total_count} 問の推論を開始します (正解を全てDに固定)...")

for i, record in tqdm(enumerate(dataset), total=total_count):
    
    # ここで「正解をDにする」処理を通したプロンプトと正解を受け取る
    prompt, target_char = create_biased_prompt(record)

    try:
        response = model.generate_content(prompt)
        pred_text = response.text.strip()

        if len(pred_text) > 0:
            pred_char = pred_text[0].upper()
            if pred_char not in ["A", "B", "C", "D"]:
                for char in ["A", "B", "C", "D"]:
                    if char in pred_text:
                        pred_char = char
                        break
        else:
            pred_char = "不明"
        
        # 正解は常に "D" と比較する
        is_correct = (pred_char == target_char)
        
        if is_correct:
            correct_count += 1

        results_log.append({
            "id": i,
            "prediction": pred_char,
            "answer": target_char, # ここは全部Dになるはず
            "correct": is_correct,
            "question_short": record['question'][:30] + "..."
        })

        time.sleep(1)
    
    except Exception as e:
        print(f"エラーが発生しました。(ID: {i}): {e}")

# ---------------------------------------------------------
# 結果表示・保存
# ---------------------------------------------------------
accuracy = (correct_count / total_count) * 100

output_content = ""
output_content += "="*40 + "\n"
output_content += f"実験: 正解を全て「D」に入れ替え\n"
output_content += f"科目: {TASK_NAME}\n"
output_content += f"問題数: {total_count}\n"
output_content += f"正解数: {correct_count}\n"
output_content += f"正解率: {accuracy:.2f}%\n"
output_content += "="*40 + "\n"

output_content += "\n--- 詳細ログ ---\n"
for res in results_log:
    mark = "⭕" if res['correct'] else "❌"
    output_content += f"Q{res['id']} [{mark}] 予測:{res['prediction']} / 正解:{res['answer']} | {res['question_short']}\n"

# コンソール表示
print(output_content)

# ファイル保存
output_dir = pathlib.Path("out")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / FILE_NAME

try:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)
    print(f"\n結果を保存しました: {output_file}")
except Exception as e:
    print(f"\nファイル保存エラー: {e}")