# src/train.py
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# ----------------------------
# Конфигурация
# ----------------------------
MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
MODEL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models", "rubert_news_classifier")

# ----------------------------
# Загрузка данных
# ----------------------------
print("📂 Загружаем обработанные данные...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

unique_topics = sorted(train_df['topic'].unique())
label2id = {topic: idx for idx, topic in enumerate(unique_topics)}
id2label = {idx: topic for topic, idx in label2id.items()}

print(f"📚 Классы: {unique_topics}")

train_df['label'] = train_df['topic'].map(label2id)
val_df['label'] = val_df['topic'].map(label2id)

# ----------------------------
# Подготовка датасетов
# ----------------------------
print("⚙️ Преобразуем в Hugging Face Dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=MAX_LENGTH)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ----------------------------
# Модель
# ----------------------------
print(f"🦾 Загружаем модель {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ----------------------------
# Метрики
# ----------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# ----------------------------
# Обучение без evaluation_strategy
# ----------------------------
print("🚀 Начинаем обучение...")

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=1,  # Обучаем по 1 эпохе за раз
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=100,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# Цикл по эпохам с ручной валидацией
for epoch in range(int(EPOCHS)):
    print(f"\n🔹 Эпоха {epoch + 1}/{EPOCHS}")
    trainer.train()
    print("🧪 Валидация...")
    metrics = trainer.evaluate()
    print(f"✅ Результаты: Accuracy = {metrics['eval_accuracy']:.4f}, F1 = {metrics['eval_f1_macro']:.4f}")

# Сохраняем финальную модель
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print(f"\n✅ Модель сохранена в: {MODEL_OUTPUT_DIR}")
print("🎯 Обучение завершено!")