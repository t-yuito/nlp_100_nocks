import os
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai

FILE_NAME = "answer_history_044.txt"

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")

question_text = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
"""

prompt = f"""
以下の問いかけに対する応答を作成してください。
思考プロセスもあわせて記述し、最終的な正解を導き出してください。

問いかけ：
{question_text}
"""

try:
    response = model.generate_content(prompt)
    answer_text = response.text

    print("\n" + "="*40)
    print(answer_text)
    print("="*40)

    output_dir = pathlib.Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / FILE_NAME

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(answer_text)

except Exception as e:
    print(f"エラーが発生しました： {e}")