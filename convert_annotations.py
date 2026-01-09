"""
Конвертер CSV из формата Владимира в формат для prepare_dataset.py

Преобразует:
- filename,text → image_path,transcription
- test_000.jpg → raw/test_000.jpg
"""
import pandas as pd

def convert_annotations(
    input_csv: str = "data/annotated/annotations.csv",
    output_csv: str = "data/raw_samples.csv"
):
    """
    Конвертирует CSV из формата (filename, text) 
    в формат (image_path, transcription) с добавлением raw/ префикса
    """
    print(f"[*] Читаю {input_csv}...")
    
    # Читаем исходный CSV
    df = pd.read_csv(input_csv, encoding='utf-8')
    
    # Проверяем наличие нужных колонок
    if 'filename' in df.columns and 'text' in df.columns:
        # Переименовываем колонки
        df = df.rename(columns={'filename': 'image_path', 'text': 'transcription'})
    else:
        print(f"[!] Найдены колонки: {list(df.columns)}")
        # Если колонки уже правильные, используем как есть
        if 'image_path' not in df.columns or 'transcription' not in df.columns:
            print("[ERROR] Ошибка: неправильный формат CSV")
            return
    
    # Добавляем raw/ к путям (если ещё не добавлено)
    if not df['image_path'].str.startswith('raw/').all():
        df['image_path'] = 'raw/' + df['image_path']
    
    # Сохраняем
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"[OK] Конвертация завершена!")
    print(f"   Входной файл: {input_csv}")
    print(f"   Выходной файл: {output_csv}")
    print(f"   Обработано записей: {len(df)}")
    print(f"\n[*] Первые 3 строки результата:")
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    convert_annotations()
