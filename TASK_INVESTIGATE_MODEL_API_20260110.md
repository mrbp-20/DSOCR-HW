# 🔬 ЗАДАНИЕ: ИССЛЕДОВАНИЕ РЕАЛЬНОГО API DeepSeek-OCR

**Автор:** Семён (Tech Lead)  
**Дата:** 2026-01-10, 17:36 MSK  
**Приоритет:** 🔥 CRITICAL (фундаментальное открытие!)  
**Срок:** 30-40 минут (исследование)  
**Исполнитель:** Николай (Senior ML Engineer)  
**Связано с:** TASK_FIX_PEFT_WRAPPER_20260110.md

---

## 🎯 КОНТЕКСТ

Николай, **ОТЛИЧНАЯ работа с диагностикой!** 🏆

Wrapper работает, `decoder_input_ids` больше не проблема. НО новая ошибка открыла **фундаментальную проблему**:

```
TypeError: DeepseekOCRForCausalLM.forward() got an unexpected keyword argument 'pixel_values'
```

## 💡 КЛЮЧЕВОЕ ОТКРЫТИЕ

**DeepSeek-OCR принимает изображения НЕ через `pixel_values`!**

Это **НЕ стандартная Vision2Seq модель** (типа BLIP, LLaVA). У неё **кастомный формат данных**.

---

## 🔍 ЧТО Я НАШЁЛ (подсказки для старта)

### 1. Официальная документация HuggingFace

**Ссылка:** https://huggingface.co/deepseek-ai/DeepSeek-OCR

**Ключевая информация:**

```python
# Пример использования из офиц. репо
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}  # ← НЕ pixel_values!
    }
]
```

**Вопрос для тебя:**
- Как `multi_modal_data` конвертируется в формат для `model.forward()`?
- Что происходит внутри процессора при `prepare_inputs`?

---

### 2. GitHub репозиторий DeepSeek-OCR

**Ссылка:** https://github.com/deepseek-ai/DeepSeek-OCR

**Что исследовать:**
- [ ] Найти примеры **training** (не inference!)
- [ ] Посмотреть, как они создают `batch` для обучения
- [ ] Какие ключи (`keys`) в словаре batch для `model.forward()`?

**Подсказка:** Ищи файлы типа `train.py`, `finetune.py`, `examples/training/`

---

### 3. Формат данных для обучения

**Ссылка (Technical Report):** https://pkulium.github.io/DeepOCR_website/

**Цитата из документации:**
> "Data Format: Image input: **[pixel_values, images_crop, images_spatial_crop]**"

**Вопросы:**
1. Что такое `images_crop` и `images_spatial_crop`?
2. Как их создать из PIL Image?
3. Есть ли в `VLChatProcessor` метод для их генерации?

---

### 4. vLLM Recipe для DeepSeek-OCR

**Ссылка:** https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html

**Что проверить:**
- Как vLLM передаёт изображения в модель?
- Есть ли там примеры **batch processing**?

---

### 5. DataCamp Tutorial

**Ссылка:** https://www.datacamp.com/tutorial/deepseek-ocr-hands-on-guide

**Что искать:**
- Раздел про training/fine-tuning
- Примеры создания `batch` для обучения
- Как processor обрабатывает изображения

---

## 🎯 ЗАДАЧА ДЛЯ ТЕБЯ

### Цель исследования

**Найти ТОЧНЫЙ формат данных для `DeepseekOCRForCausalLM.forward()` при обучении.**

### Что нужно выяснить

1. **Какие параметры принимает `model.forward()`?**
   - Точные названия: `pixel_values`? `images`? `input_images`?
   - Формат каждого параметра: shape, dtype
   - Обязательные vs опциональные

2. **Как processor создаёт эти данные?**
   - Вызов: `processor(images=..., text=..., return_tensors="pt")`
   - Что возвращает? Какие ключи в словаре?
   - Нужны ли `images_crop`, `images_spatial_crop`?

3. **Как выглядит правильный batch для training?**
   - Структура словаря для `model(**batch)`
   - Пример реального batch из офиц. кода

4. **Нужен ли кастомный DataCollator?**
   - Стандартный `DataCollatorForSeq2Seq` подходит?
   - Или нужен специфичный для DeepSeek-OCR?

---

## 📋 ПОШАГОВАЯ ИНСТРУКЦИЯ

### Шаг 1: Исследовать HuggingFace репо модели

```powershell
# Клонировать примеры (если есть)
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
cd DeepSeek-OCR
# Искать файлы с примерами обучения
```

**ИЛИ** просто читать код на сайте HuggingFace.

**Что искать:**
- `README.md` — раздел "Training" или "Fine-tuning"
- Файлы `.py` с примерами
- Issues/Discussions о fine-tuning

---

### Шаг 2: Тестовый скрипт для исследования API

Создай `scripts/test_model_api.py`:

```python
"""
Тестовый скрипт для исследования API DeepSeek-OCR.
Цель: понять, какие параметры принимает model.forward().
"""

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, VLChatProcessor

# Загрузить модель и процессор
model_name = "deepseek-ai/DeepSeek-OCR"
print(f"Загрузка модели: {model_name}")

processor = VLChatProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"  # Для тестирования на CPU
)

print(f"\n✅ Модель загружена: {type(model)}")
print(f"✅ Процессор загружен: {type(processor)}")

# Создать тестовое изображение
image = Image.new("RGB", (224, 224), color="white")
text = "Test OCR prompt"

# ИССЛЕДОВАНИЕ 1: Что возвращает processor?
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 1: Вызов processor")
print("="*80)

inputs = processor(
    images=image,
    text=text,
    return_tensors="pt"
)

print(f"\nВозвращённые ключи: {list(inputs.keys())}")
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: type={type(value)}")

# ИССЛЕДОВАНИЕ 2: Какие параметры принимает model.forward()?
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 2: Signature model.forward()")
print("="*80)

import inspect
forward_signature = inspect.signature(model.forward)
print(f"\nПараметры model.forward():")
for param_name, param in forward_signature.parameters.items():
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")

# ИССЛЕДОВАНИЕ 3: Попытка вызвать model.forward()
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 3: Тестовый вызов model.forward()")
print("="*80)

try:
    # Попробовать передать все ключи из processor
    outputs = model(**inputs)
    print("✅ model.forward() УСПЕШЕН!")
    print(f"Outputs type: {type(outputs)}")
    print(f"Outputs keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'N/A'}")
except TypeError as e:
    print(f"❌ TypeError: {e}")
    print("\nПробуем передать по одному параметру...")
    
    # Попробовать разные комбинации
    test_cases = [
        {"pixel_values": inputs.get("pixel_values")},
        {"images": inputs.get("pixel_values")},
        {"input_ids": inputs.get("input_ids")},
        {"pixel_values": inputs.get("pixel_values"), "input_ids": inputs.get("input_ids")},
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n  Тест {i}: передаём {list(test_input.keys())}")
        try:
            outputs = model(**test_input)
            print(f"    ✅ РАБОТАЕТ!")
            break
        except Exception as e:
            print(f"    ❌ {type(e).__name__}: {e}")

# ИССЛЕДОВАНИЕ 4: Проверить processor.image_processor
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 4: processor.image_processor")
print("="*80)

if hasattr(processor, 'image_processor'):
    print(f"\nimage_processor type: {type(processor.image_processor)}")
    print(f"Методы image_processor:")
    for attr in dir(processor.image_processor):
        if not attr.startswith('_'):
            print(f"  - {attr}")

print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
print("="*80)
```

**Запустить:**

```powershell
cd C:\DSOCR-HW
.\.venv\Scripts\Activate.ps1
python scripts/test_model_api.py > test_model_api_output.txt 2>&1
```

**Результат:** файл `test_model_api_output.txt` с полной информацией об API.

---

### Шаг 3: Исследовать GitHub Issues

**Ссылки для поиска:**
- https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Ищи issues с ключевыми словами: "training", "fine-tune", "pixel_values", "batch"

**Вопросы для поиска:**
- Как другие файнтюнят DeepSeek-OCR?
- Какие проблемы они встречали?
- Есть ли рабочие примеры кода?

---

### Шаг 4: Проверить processor подробнее

Добавь в `test_model_api.py`:

```python
# ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ: prepare_inputs_for_generation
if hasattr(model, 'prepare_inputs_for_generation'):
    print("\n" + "="*80)
    print("prepare_inputs_for_generation signature")
    print("="*80)
    sig = inspect.signature(model.prepare_inputs_for_generation)
    for param_name, param in sig.parameters.items():
        print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
```

---

## 🎯 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ

### Отчёт `REPORT_MODEL_API_INVESTIGATION_20260110.md`

**Создай отчёт со следующей структурой:**

```markdown
# 🔬 ОТЧЁТ: ИССЛЕДОВАНИЕ API DeepSeek-OCR

**Дата:** 2026-01-10  
**Исполнитель:** Николай  

---

## 1. ПАРАМЕТРЫ model.forward()

**Найденные параметры:**
- `parameter_1`: описание, тип, shape
- `parameter_2`: описание, тип, shape
- ...

**Обязательные:** [список]  
**Опциональные:** [список]

---

## 2. ВЫВОД processor()

**Ключи в outputs:**
- `key_1`: shape, dtype, пример значения
- `key_2`: shape, dtype, пример значения
- ...

**Совпадают ли с параметрами model.forward()?** Да/Нет  
**Какие ключи не нужны для forward()?** [список]

---

## 3. ПРАВИЛЬНЫЙ ФОРМАТ BATCH

**Структура batch для training:**

```python
batch = {
    "key_1": ...,  # описание
    "key_2": ...,  # описание
    # ...
}
```

**Откуда взята информация:** [ссылка на источник]

---

## 4. DATA COLLATOR

**Нужен ли кастомный DataCollator?** Да/Нет

**Если да, что он должен делать:**
1. ...
2. ...

**Если нет, почему стандартный подходит:**
...

---

## 5. ПРИМЕРЫ ИЗ ОФИЦ. РЕПО

**Найденные примеры обучения:**
- [Ссылка 1] — краткое описание
- [Ссылка 2] — краткое описание

**Ключевые находки из примеров:**
...

---

## 6. РЕКОМЕНДАЦИИ

**Что нужно изменить в нашем коде:**

1. **В data_collator.py:**
   - Изменение 1
   - Изменение 2

2. **В model_wrapper.py:**
   - Изменение 1 (если нужно)

3. **В trainer.py:**
   - Изменение 1 (если нужно)

**Приоритет изменений:** [1 - критично, 2 - важно, 3 - опционально]

---

## 7. ВОПРОСЫ К СЕМЁНУ

(Если что-то не понятно из исследования)

1. Вопрос 1?
2. Вопрос 2?

---

## ПРИЛОЖЕНИЯ

### Лог test_model_api.py

```
[вставить вывод скрипта]
```

### Скриншоты (если есть)

[вставить]
```

---

## 📚 РЕСУРСЫ ДЛЯ ИССЛЕДОВАНИЯ

### Обязательные

1. ✅ **HuggingFace Model Card**  
   https://huggingface.co/deepseek-ai/DeepSeek-OCR
   
2. ✅ **GitHub Repository**  
   https://github.com/deepseek-ai/DeepSeek-OCR
   
3. ✅ **Technical Report**  
   https://pkulium.github.io/DeepOCR_website/

### Дополнительные

4. **vLLM Recipe**  
   https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html
   
5. **DataCamp Tutorial**  
   https://www.datacamp.com/tutorial/deepseek-ocr-hands-on-guide
   
6. **Google Vertex AI Docs**  
   https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/deepseek/deepseek-ocr

### Полезные поисковые запросы

- "DeepSeek-OCR fine-tuning example"
- "DeepseekOCRForCausalLM forward parameters"
- "DeepSeek-OCR training batch format"
- "VLChatProcessor images_crop images_spatial_crop"

---

## ⏱️ ТАЙМИНГ

- **Исследование документации:** 10 минут
- **Запуск test_model_api.py:** 5 минут
- **Поиск примеров в GitHub:** 10 минут
- **Написание отчёта:** 10-15 минут

**Общее время:** ~40 минут

---

## 🎯 КРИТЕРИИ УСПЕХА

### Минимальный результат ✅

- [ ] Найдены **точные названия параметров** для `model.forward()`
- [ ] Известен **формат batch** для training
- [ ] Понятно, **как processor создаёт** эти данные

### Идеальный результат 🏆

- [ ] Найден **рабочий пример** fine-tuning DeepSeek-OCR
- [ ] Написан **готовый код** для нового DataCollator
- [ ] Понятны **все особенности** модели (crops, spatial crops, etc.)

---

## 💬 СВЯЗЬ

**Если застрял:**
1. Отпиши в отчёт, что успел найти
2. Опиши, где застрял
3. Задай вопросы Семёну

**Если всё нашёл раньше 40 минут:**
🎉 Ты красавчик! Пиши отчёт и коммить!

---

## 🚀 ПОСЛЕ ИССЛЕДОВАНИЯ

Когда отчёт готов:

1. Закоммитить всё:
```powershell
git add .
git commit -m "docs: add model API investigation report"
git push
```

2. Сообщить Семёну — он **сразу** даст финальное ТЗ на основе твоих находок!

---

**Николай, это КРИТИЧЕСКИ важное исследование!** 🔬

От него зависит **весь дальнейший путь** проекта. Ты единственный, кто может сейчас разобраться в этом — твой опыт senior engineer'а здесь **ключевой**.

**Удачи в исследовании!** 🏆

---

**С уважением и уверенностью в твоих способностях,**  
**Семён (Tech Lead)** 🎯

P.S. Помни: "The best debugging tool is your brain, but the best research tool is official documentation + your curiosity!" 🧠📚

P.P.S. Если найдёшь, что DeepSeek-OCR требует каких-то экзотических crops — не пугайся, мы напишем код для их генерации. Главное — **понять формат**! 💪
