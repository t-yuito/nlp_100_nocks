import os
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai

FILE_NAME = "answer_history_045.txt"

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")

chat = model.start_chat(history=[])

question_1 = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
"""

response_1 = chat.send_message(question_1)
print(f"{response_1.text.strip()[:50]}...")

question_2 = """
さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
思考ミスを防ぐため、乗車する駅と降車する駅の間にある駅名を順序立てて列挙し、正確に数えて回答してください。
"""

response_2 = chat.send_message(question_2)

print("\n" + "="*40)
print(response_2.text)
print("="*40)

output_dir = pathlib.Path("out")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / FILE_NAME

output_content = f"""
問いかけ
{question_2.strip()}
応答
{response_2.text}
"""

try:
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(output_content)
except Exception as e:
    print(f"ファイル保存エラー： {e}")