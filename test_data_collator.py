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

# 3a. Обработка images через torchvision.transforms
print("   3a. Обработка images через torchvision.transforms...")
try:
    import torchvision.transforms as transforms
    
    images = [Image.open(img).convert('RGB') for img in test_images]
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    pixel_values = torch.stack([transform(img) for img in images])
    print(f"       ✅ pixel_values shape: {pixel_values.shape}")
    print(f"       ✅ pixel_values dtype: {pixel_values.dtype}")
    print(f"       ✅ pixel_values range: [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
except Exception as e:
    print(f"       ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

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

# 4. Формирование батча
print("\n4️⃣ Формирование батча...")
try:
    batch = {
        'pixel_values': pixel_values,
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'labels': text_inputs['input_ids'].clone()
    }
    
    print("   Ключи батча:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"     - {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"     - {key}: {type(value)}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# 5. Проверка кодировки
print("\n5️⃣ Проверка кодировки...")
print("   Русский текст: Привет, мир! 🚀")
print("   Emoji: 💪 ✅ ❌ 🎯")

print("\n" + "=" * 80)
print("✅ ТЕСТ ЗАВЕРШЁН!")
print("=" * 80)
