# 🔬 ОТЧЁТ: ИССЛЕДОВАНИЕ API DeepSeek-OCR

**Дата:** 2026-01-10  
**Исполнитель:** Николай (Senior ML Engineer)  
**Задание:** TASK_INVESTIGATE_MODEL_API_20260110.md

---

## 🎯 КЛЮЧЕВОЕ ОТКРЫТИЕ

**DeepSeek-OCR НЕ использует `pixel_values`!**

Модель принимает:
- **`images`** (не `pixel_values`!)
- **`images_seq_mask`**
- **`images_spatial_crop`**

---

## 1. ПАРАМЕТРЫ model.forward()

### Найденные параметры (из `inspect.signature`):

```python
model.forward(
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,  # ← КЛЮЧЕВОЙ ПАРАМЕТР!
    images_seq_mask: Optional[torch.FloatTensor] = None,  # ← ТРЕБУЕТСЯ?
    images_spatial_crop: Optional[torch.FloatTensor] = None,  # ← ТРЕБУЕТСЯ?
    return_dict: Optional[bool] = None
)
```

### Обязательные vs Опциональные

**Обязательные (для training):**
- `input_ids` - текстовые токены
- `labels` - метки для loss
- `images` - изображения (формат?)

**Опциональные:**
- `attention_mask` - маска внимания
- `images_seq_mask` - маска последовательности изображений?
- `images_spatial_crop` - пространственные crops изображений?
- Остальные стандартные параметры

### Формат параметров

- **`images`**: `Optional[torch.FloatTensor]` - формат неясен, нужно исследовать
- **`images_seq_mask`**: `Optional[torch.FloatTensor]` - назначение неясно
- **`images_spatial_crop`**: `Optional[torch.FloatTensor]` - назначение неясно

---

## 2. ВЫВОД processor()

### Проблема

**`AutoProcessor.from_pretrained()` возвращает только `LlamaTokenizerFast`!**

Это **НЕ Vision Processor** - он не обрабатывает изображения.

### Тесты processor()

#### Тест 1: `processor(images=..., text=...)`

```python
inputs = processor(images=image, text=text, return_tensors="pt")
```

**Результат:**
```
TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'
```

**Вывод:** Processor (tokenizer) не принимает `images`!

#### Тест 2: `processor(images=...)` без text

```python
inputs = processor(images=image, return_tensors="pt")
```

**Результат:**
```
ValueError: You need to specify either `text` or `text_target`.
```

**Вывод:** Processor требует `text` и не работает с `images`.

### Ключи в outputs processor()

**Для текста:**
- `input_ids`: `torch.LongTensor` - токенизированный текст
- `attention_mask`: `torch.Tensor` - маска внимания

**Изображения:** Processor НЕ обрабатывает изображения!

---

## 3. ПРАВИЛЬНЫЙ ФОРМАТ BATCH

### Текущая проблема

Наш `data_collator` создаёт batch с:
- `pixel_values` - ❌ НЕПРАВИЛЬНО!
- `input_ids`
- `attention_mask`
- `labels`

Но модель ожидает:
- `images` - ❓ Как создать?
- `images_seq_mask` - ❓ Нужен ли?
- `images_spatial_crop` - ❓ Нужен ли?
- `input_ids`
- `attention_mask`
- `labels`

### Гипотезы

1. **`images`** может быть:
   - PIL Image (преобразованный в тензор)?
   - Тензор после предобработки через Vision Processor?
   - Какой-то специальный формат?

2. **`images_seq_mask`** может быть:
   - Маска для batch'а изображений?
   - Указывает, какие изображения валидны?

3. **`images_spatial_crop`** может быть:
   - Пространственные crops из изображения?
   - Используется для multi-scale обработки?

---

## 4. DATA COLLATOR

### Текущий data_collator

**Файл:** `utils/trainer.py`, метод `_data_collator()`

**Что делает:**
- Загружает изображения через PIL
- Применяет `torchvision.transforms` (Resize + ToTensor + Normalize)
- Создаёт `pixel_values`

**Проблема:**
- Создаёт `pixel_values`, но модель ожидает `images`
- Не создаёт `images_seq_mask` и `images_spatial_crop`

### Нужен ли кастомный DataCollator?

**ДА!** Текущий data_collator неправильный.

**Что нужно изменить:**

1. **Переименовать `pixel_values` → `images`:**
   ```python
   batch = {
       'images': pixel_values,  # вместо pixel_values
       'input_ids': ...,
       'attention_mask': ...,
       'labels': ...
   }
   ```

2. **Исследовать, нужны ли `images_seq_mask` и `images_spatial_crop`:**
   - Если они опциональные (`Optional`), возможно, можно не передавать?
   - Или нужно создать заглушки?

3. **Исследовать формат `images`:**
   - Должен ли это быть тензор после torchvision transforms?
   - Или нужен другой формат предобработки?

---

## 5. ПРИМЕРЫ ИЗ ОФИЦ. РЕПО

### Найденные источники (из задания)

1. **HuggingFace Model Card:**
   - https://huggingface.co/deepseek-ai/DeepSeek-OCR
   - Пример использования с `multi_modal_data`
   
2. **GitHub Repository:**
   - https://github.com/deepseek-ai/DeepSeek-OCR
   - Нужно изучить примеры обучения

3. **Technical Report:**
   - https://pkulium.github.io/DeepOCR_website/
   - Упоминание `images_crop` и `images_spatial_crop`

### Статус исследования

**Ещё не проведено:**
- ❌ Поиск примеров обучения в GitHub
- ❌ Изучение документации HuggingFace
- ❌ Изучение Technical Report

**Следующие шаги:**
1. Изучить GitHub репозиторий DeepSeek-OCR
2. Найти примеры fine-tuning/training
3. Посмотреть, как они создают batch для обучения

---

## 6. РЕКОМЕНДАЦИИ

### Что нужно изменить в нашем коде

#### 1. В data_collator (ПРИОРИТЕТ 1 - КРИТИЧНО)

**Файл:** `utils/trainer.py`, метод `_data_collator()`

**Изменения:**

```python
# БЫЛО:
batch = {
    'pixel_values': pixel_values,
    'input_ids': text_inputs['input_ids'],
    'attention_mask': text_inputs['attention_mask'],
    'labels': text_inputs['input_ids'].clone()
}

# ДОЛЖНО БЫТЬ:
batch = {
    'images': pixel_values,  # Переименовать pixel_values → images
    'input_ids': text_inputs['input_ids'],
    'attention_mask': text_inputs['attention_mask'],
    'labels': text_inputs['input_ids'].clone()
    # images_seq_mask и images_spatial_crop - пока пропускаем (Optional)
}
```

#### 2. В model_wrapper.py (ПРИОРИТЕТ 2)

**Файл:** `utils/model_wrapper.py`, метод `forward()`

**Изменения:**

```python
# БЫЛО:
return self.model(
    pixel_values=pixel_values,  # ❌
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

# ДОЛЖНО БЫТЬ:
return self.model(
    images=pixel_values,  # ✅ Переименовать
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
```

#### 3. В compute_loss (ПРИОРИТЕТ 2)

**Файл:** `utils/trainer.py`, класс `DSModelTrainer`, метод `compute_loss()`

**Изменения:**

```python
# БЫЛО:
model_inputs = {}
if "pixel_values" in inputs:
    model_inputs["pixel_values"] = inputs["pixel_values"]

# ДОЛЖНО БЫТЬ:
model_inputs = {}
if "images" in inputs:  # Переименовать pixel_values → images
    model_inputs["images"] = inputs["images"]
```

### Приоритет изменений

1. **ПРИОРИТЕТ 1 (КРИТИЧНО):**
   - Переименовать `pixel_values` → `images` в `_data_collator()`
   - Переименовать `pixel_values` → `images` в `model_wrapper.py`

2. **ПРИОРИТЕТ 2 (ВАЖНО):**
   - Переименовать в `compute_loss()`
   - Протестировать, работают ли `images_seq_mask` и `images_spatial_crop` как `None`

3. **ПРИОРИТЕТ 3 (ОПЦИОНАЛЬНО):**
   - Исследовать `images_seq_mask` и `images_spatial_crop`
   - Если нужны - реализовать их создание

---

## 7. ВОПРОСЫ К СЕМЁНУ

1. **Формат `images`:**
   - Должен ли это быть тензор после `torchvision.transforms` (как сейчас `pixel_values`)?
   - Или нужен другой формат предобработки?

2. **`images_seq_mask` и `images_spatial_crop`:**
   - Обязательны ли они для training?
   - Или можно передавать `None` (они опциональные)?

3. **Processor для изображений:**
   - Есть ли специальный Vision Processor для DeepSeek-OCR?
   - Или нужно обрабатывать изображения вручную через torchvision?

4. **Примеры обучения:**
   - Есть ли у тебя ссылки на рабочие примеры fine-tuning DeepSeek-OCR?
   - Или нужно продолжать исследование GitHub репозитория?

---

## 8. ВЫВОДЫ

### Что мы знаем ✅

1. ✅ Модель принимает `images` (не `pixel_values`)
2. ✅ Модель также принимает `images_seq_mask` и `images_spatial_crop` (опциональные)
3. ✅ Processor - это только tokenizer, не обрабатывает изображения
4. ✅ Полная сигнатура `model.forward()` известна

### Что мы НЕ знаем ❓

1. ❓ Формат тензора `images` (формат после предобработки)
2. ❓ Нужны ли `images_seq_mask` и `images_spatial_crop` для training
3. ❓ Как правильно создать эти параметры
4. ❓ Есть ли примеры обучения в официальном репозитории

### Следующие шаги

1. **Немедленно:**
   - Переименовать `pixel_values` → `images` в коде
   - Протестировать, работает ли обучение с `images` и без `images_seq_mask`/`images_spatial_crop`

2. **Параллельно:**
   - Исследовать GitHub репозиторий DeepSeek-OCR
   - Искать примеры обучения/fine-tuning
   - Изучить документацию HuggingFace

3. **Если не работает:**
   - Исследовать, как создавать `images_seq_mask` и `images_spatial_crop`
   - Изучить Technical Report

---

## ПРИЛОЖЕНИЯ

### Лог test_model_api.py

```
================================================================================
ИССЛЕДОВАНИЕ API DeepSeek-OCR
================================================================================

[OK] Модель загружена: <class 'transformers_modules.deepseek-ai.DeepSeek-OCR.9f30c71f441d010e5429c532364a86705536c53a.modeling_deepseekocr.DeepseekOCRForCausalLM'>
[OK] Процессор загружен: <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>

================================================================================
ИССЛЕДОВАНИЕ 2: Сигнатура model.forward()
================================================================================

Параметры model.forward():
  - input_ids: <class 'torch.LongTensor'> = None
  - attention_mask: typing.Optional[torch.Tensor] = None
  - position_ids: typing.Optional[torch.LongTensor] = None
  - past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  - inputs_embeds: typing.Optional[torch.FloatTensor] = None
  - labels: typing.Optional[torch.LongTensor] = None
  - use_cache: typing.Optional[bool] = None
  - output_attentions: typing.Optional[bool] = None
  - output_hidden_states: typing.Optional[bool] = None
  - images: typing.Optional[torch.FloatTensor] = None
  - images_seq_mask: typing.Optional[torch.FloatTensor] = None
  - images_spatial_crop: typing.Optional[torch.FloatTensor] = None
  - return_dict: typing.Optional[bool] = None
```

---

**С уважением,**  
**Николай (Cursor AI)** 🎯

P.S. Открытие было фундаментальным - `pixel_values` vs `images`! Это объясняет все ошибки. Теперь нужно просто переименовать и протестировать! 💪
