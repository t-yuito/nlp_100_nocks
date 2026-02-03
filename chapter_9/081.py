import os
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-cased")

text = "The movie was full of [MASK]."

results = fill_mask(text)

top_prediction = results[0]
predicted_token = top_prediction['token_str']
score = top_prediction['score']

output_dir = 'out'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "081.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Predicted token: {predicted_token}\n")
    f.write(f"Probability score {score:.4f}\n")

print("保存完了")