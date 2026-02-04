import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/home/takeuchi/workspace/nlp_100_nocks/chapter_9/results/checkpoint-12630"
model_ckpt = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

test_sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

output_dir = "out"
import os
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "088.txt")
status = "Ready for prediction"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Model Status: {status}\n")
    header = f"{'Sentence':<50} | {'Prediction':<10} | {'Confidence'}"
    f.write(header + "\n" + "-" * 80 + "\n")
    print(header)
    print("-" * 80)

    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()
        
        label = "Positive" if pred_idx == 1 else "Negative"
        
        line = f"{text:<50} | {label:<10} | {confidence:.4f}"
        f.write(line + "\n")
        print(line)

print("保存完了")