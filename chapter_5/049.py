import os
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai

FILE_NAME = "answer_history_049.txt"

TEXT = """
吾輩は猫である。名前はまだ無い。

どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
"""

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")

print("トークンを計算")

response = model.count_tokens(TEXT)
token_count = response.total_tokens
char_count = len(TEXT)

print("\n" + "="*40)
print(f"文字数: {char_count} 文字")
print(f"トークン数 {token_count} tokens")
print(f"比率 (文字/トークン): 約 {char_count / token_count:.2f} 文字/tokens")

output_dir = pathlib.Path("out")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / FILE_NAME

output_content = f"""
対象テキスト
{TEXT}

計測結果
文字数: {char_count}
トークン数: {token_count}
"""

try:
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(output_content)
except Exception as e:
    print(f"\n ファイル保存エラー: {e}")