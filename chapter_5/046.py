import os
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai

FILE_NAME = "answer_history_046.txt"
THEME = "プログラミング学習"

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
お題「{THEME}」について、川柳（五・七・五）の案を10個作成してください。
サラリーマン川柳のように、ユーモア、自虐、あるいは共感を誘う内容にしてください。
解説は不要です。作品のみを箇条書きで出力してください。
"""

try:
    response = model.generate_content(prompt)

    print("\n" + "="*40)
    print(f"川柳 10選 (お題：{THEME})")
    print("="*40)
    print(response.text)

    output_dir = pathlib.Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / FILE_NAME

    output_content = f"""
    お題
    {THEME}
    生成された川柳
    {response.text}
    """

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(output_content)
except Exception as e:
    print(f"エラーが発生しました： {e}")