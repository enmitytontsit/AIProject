# src/predict.py
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Путь к модели ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models", "rubert_news_classifier")

# === Загрузка ===
print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_topic(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax().item()
    return model.config.id2label[pred_id]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py \"Текст статьи...\"")
        print("\nПримеры:")
        examples = [
            "Президент выступил с ежегодным посланием.",
            "ЦБ повысил ключевую ставку до 16%.",
            "ЦСКА обыграл Зенит со счётом 2:1.",
            "Выставка современного искусства открылась в Третьяковке."
        ]
        for ex in examples:
            topic = predict_topic(ex)
            print(f"  → \"{ex[:50]}...\" → {topic}")
    else:
        text = " ".join(sys.argv[1:])
        topic = predict_topic(text)
        print(f"Тематика: {topic}")