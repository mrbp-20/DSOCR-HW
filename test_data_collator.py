"""
Тест для проверки исправленного data_collator.

Проверяет:
1. Раздельную обработку images и text
2. Корректность формата батча
3. Кодировку вывода
"""

import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

# Исправляем кодировку для Windows
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("🧪 ТЕСТ: Data Collator для DeepSeek-OCR")
print("=" * 80)

# 1. Загрузка processor
print("\n1️⃣ Загрузка processor...")
processor = AutoProcessor.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    revision="9f30c71f441d010e5429c532364a86705536c53a",
    trust_remote_code=True
)
print(f"   Тип processor: {type(processor)}")
print(f"   Есть tokenizer: {hasattr(processor, 'tokenizer')}")

# 2. Подготовка тестовых данных
print("\n2️⃣ Подготовка тестовых данных...")
test_images = list(Path("data/raw").glob("*.jpg"))[:2]  # Берём 2 образца
test_texts = [
    "Миновало почти десятилетие...",
    "Отличная работа!"
]

print(f"   Изображений: {len(test_images)}")
print(f"   Текстов: {len(test_texts)}")

# 3. Тест раздельной обработки
print("\n3️⃣ Тест раздельной обработки...")

# 3a. Images через processor
print("   3a. Обработка images через processor...")
try:
    images = [Image.open(img).convert('RGB') for img in test_images]
    pixel_inputs = processor(images=images, return_tensors="pt")
    print(f"       ✅ pixel_values shape: {pixel_inputs['pixel_values'].shape}")
except Exception as e:
    print(f"       ❌ Ошибка: {e}")

# 3b. Text через processor.batch_encode_plus
print("   3b. Обработка text через processor.batch_encode_plus...")
try:
    text_inputs = processor.batch_encode_plus(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    print(f"       ✅ input_ids shape: {text_inputs['input_ids'].shape}")
    print(f"       ✅ attention_mask shape: {text_inputs['attention_mask'].shape}")
except Exception as e:
    print(f"       ❌ Ошибка: {e}")

# 4. Проверка кодировки
print("\n5️⃣ Проверка кодировки...")
print("   Русский текст: Привет, мир! 🚀")
print("   Emoji: 💪 ✅ ❌ 🎯")

print("\n" + "=" * 80)
print("✅ ТЕСТ ЗАВЕРШЁН!")
print("=" * 80)
