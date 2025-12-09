# src/prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === Новое: маппинг исходных категорий на обобщённые темы ===
CATEGORY_MAPPING = {
    # Политика
    'Политика': 'Политика',
    'Россия': 'Политика',
    'Мир': 'Политика',

    # Экономика
    'Экономика': 'Экономика',
    'Бизнес': 'Экономика',
    'Авто': 'Экономика',
    'Недвижимость': 'Экономика',

    # Спорт
    'Спорт': 'Спорт',

    # Культура
    'Культура': 'Культура',
    'Шоу-бизнес': 'Культура',

    # Наука и технологии
    'Наука': 'Наука и технологии',
    'Технологии': 'Наука и технологии',
    'Интернет': 'Наука и технологии',
}

TARGET_CATEGORIES = ['Политика', 'Экономика', 'Спорт', 'Культура', 'Наука и технологии']


def prepare_dataset():
    input_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lenta-ru-news.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    print("🔍 Загружаем датасет...")
    df = pd.read_csv(input_path)

    # Применяем маппинг
    df['topic_mapped'] = df['topic'].map(CATEGORY_MAPPING)

    # Оставляем только нужные категории
    df = df[df['topic_mapped'].isin(TARGET_CATEGORIES)]

    # Удаляем пустые и короткие тексты
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 50]

    # Группируем по новым категориям
    df = df[['text', 'topic_mapped']].rename(columns={'topic_mapped': 'topic'})

    # Балансировка: до 5000 на класс
    df_balanced = df.groupby('topic').apply(
        lambda x: x.sample(min(len(x), 5000), random_state=42)
    ).reset_index(drop=True)

    # Разделение
    train, rest = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced['topic'], random_state=42)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest['topic'], random_state=42)

    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"✅ Готово! Используем категории: {TARGET_CATEGORIES}")
    print(f"📊 Размеры:")
    print(f"   Train: {len(train)}")
    print(f"   Val:   {len(val)}")
    print(f"   Test:  {len(test)}")


if __name__ == "__main__":
    prepare_dataset()