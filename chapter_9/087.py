import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# GPU 0番だけを使うように指定して実行
# CUDA_VISIBLE_DEVICES=0 uv run 087.py

model_ckpt = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

raw_datasets = load_dataset("glue", "sst2")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 学習の設定
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

print("学習開始")
trainer.train()

eval_results = trainer.evaluate()

output_dir = "out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "087.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}\n")

print("完了")