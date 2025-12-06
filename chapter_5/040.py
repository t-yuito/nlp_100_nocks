import os
from dotenv import load_dotenv
import google.generativeai as genai
import pathlib

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# モデル名を最新のものに変更
gemini = genai.GenerativeModel("gemini-2.5-flash")

# Zero-Shot推論: 追加の例示やヒントなしで、問題をそのまま投げる
prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

response = gemini.generate_content(prompt)
print(response.text)

OUTPUT_DIR = pathlib.Path("out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_NAME = "answer_history_040.txt"
output_path = OUTPUT_DIR / FILE_NAME

try:
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(response.text)
except Exception as e:
    print(f"エラーが発生しました:{e}")

