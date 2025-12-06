import os
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai

FILE_NAME = "answer_history_047.txt"
THEME = "プログラミング学習"

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")

print("川柳作成")

gen_prompt = f"""
お題「{THEME}」について、川柳（五・七・五）の案を10個作成してください。
番号付きの箇条書きで出力してください。
"""

try:
    senryu_response = model.generate_content(gen_prompt)
    senryu_text = senryu_response.text.strip()

    print("川柳を採点")

    judge_prompt = f"""
    あなたは厳格な川柳コンテストの審査員です。
    以下にリストアップされた「{THEME}」に関する川柳を、1つずつ10段階（10点が最高）で評価してください。

    【評価基準】
    - 共感性（エンジニアの苦労や喜びが伝わるか）
    - ユーモア（クスッと笑えるか）
    - リズム（五七五として美しいか）

    【出力フォーマット】
    各作品について以下の形式で出力してください：

    N. [点数/10] 川柳の本文
    寸評: （短いコメント）

    --- 対象の川柳 ---
    {senryu_text}
    """

    eval_response = model.generate_content(judge_prompt)
    eval_text = eval_response.text

    print("\n" + "="*40)
    print(eval_text)

    output_dir = pathlib.Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / FILE_NAME

    output_content = f"""
    生成された川柳
    {senryu_text}
    評価
    {eval_text}
    """

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(output_content)

except Exception as e:
    print(f"エラーが発生しました: {e}")