# src/evaluate.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

# === Пути (автоматически работают от любого места) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models", "rubert_news_classifier")
TEST_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "test.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"📂 Загрузка модели из: {MODEL_DIR}")
print(f"📄 Тестовые данные: {TEST_PATH}")

# === Загрузка модели ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Загрузка данных ===
test_df = pd.read_csv(TEST_PATH)
texts = test_df['text'].tolist()
true_labels = test_df['topic'].tolist()

# === Предсказания ===
def predict_batch(texts_batch):
    inputs = tokenizer(
        texts_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).cpu().numpy()

all_preds = []
batch_size = 32
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    preds = predict_batch(batch)
    all_preds.extend(preds)

# === Метрики ===
id2label = model.config.id2label
pred_labels = [id2label[p] for p in all_preds]

print("\n📊 Классификационный отчёт на TEST:")
report = classification_report(true_labels, pred_labels, output_dict=True)
print(classification_report(true_labels, pred_labels))

# === График 1: Метрики по классам ===
classes = list(report.keys())[:-3]  # exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report[c]['precision'] for c in classes]
recall = [report[c]['recall'] for c in classes]
f1 = [report[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, precision, width, label='Precision', color='#4e79a7')
ax.bar(x, recall, width, label='Recall', color='#f28e2b')
ax.bar(x + width, f1, width, label='F1-score', color='#e15759')

ax.set_xlabel('Тематические категории')
ax.set_ylabel('Значение метрики')
ax.set_title('Метрики качества по категориям (тестовая выборка, n=3000)')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "metrics_by_class.png"), dpi=300, bbox_inches='tight')
plt.show()

# === График 2: Матрица ошибок (confusion matrix) ===
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_labels, pred_labels, labels=classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.title('Матрица ошибок (Confusion Matrix)')
plt.xlabel('Предсказанная категория')
plt.ylabel('Истинная категория')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# === График 3: Общая точность по эпохам (из логов обучения) ===
# (Используем ваши реальные значения из лога)
epochs = [1, 2, 3]
val_accuracy = [0.949, 0.950, 0.948]
val_f1 = [0.9487, 0.9497, 0.9474]
train_loss = [0.217, 0.132, 0.093]

fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:red'
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(epochs, train_loss, color=color, marker='o', label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Валидационная точность / F1', color=color)
ax2.plot(epochs, val_accuracy, color='blue', marker='s', label='Accuracy')
ax2.plot(epochs, val_f1, color='green', marker='^', label='F1-score')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Динамика обучения (rubert-tiny2)')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.savefig(os.path.join(RESULTS_DIR, "training_dynamics.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Все графики сохранены в: {RESULTS_DIR}")