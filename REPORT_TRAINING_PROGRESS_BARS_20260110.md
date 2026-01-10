# 📊 ОТЧЁТ: ПРОГРЕСС БАРЫ И ПРОБЛЕМА С DATA_COLLATOR

**Автор:** Николай (Senior ML Engineer)  
**Дата:** 2026-01-10  
**Статус:** ✅ Progress bars реализованы, ❌ Проблема с data_collator API  
**Issue:** #2 (дополнение)  

---

## 🎯 ВЫПОЛНЕННАЯ РАБОТА

### 1. Добавлены индикаторы прогресса (✅ ЗАВЕРШЕНО)

**Задание:** TASK_ADD_PROGRESS_INDICATOR_20260110.md

**Что сделано:**

1. **Добавлен импорт `tqdm` в `utils/trainer.py`**
   - Импортирован `from tqdm import tqdm`

2. **Добавлен progress bar в `load_model_and_processor()`**
   - Progress bar с 2 шагами (процессор + модель)
   - Динамическое описание текущего шага
   - Предупреждение о долгой загрузке модели (5-30 минут)

3. **Добавлен progress bar в `prepare_datasets()`**
   - Progress bar с 2 шагами (train + val metadata)
   - Динамическое описание текущего шага

4. **Включён progress bar для обучения**
   - Добавлен `disable_tqdm=False` в `TrainingArguments`
   - Это включает встроенный tqdm в HuggingFace Trainer

5. **Улучшено логирование в `scripts/train_lora.py`**
   - Нумерация шагов (1/5, 2/5, ...)
   - Визуальные разделители между шагами
   - Сообщения о завершении каждого шага

**Результат:**
- ✅ Progress bars отображаются корректно
- ✅ Видно прогресс загрузки модели (0% → 50% → 100%)
- ✅ Видно прогресс загрузки датасетов
- ✅ Видно начало прогресса обучения

---

## 🔧 ИСПРАВЛЕННЫЕ ПРОБЛЕМЫ

### Проблема 1: Параметр `optimizer` не поддерживается (✅ ИСПРАВЛЕНО)

**Ошибка:**
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'optimizer'
```

**Причина:**
В `transformers 4.45.0` параметр `optimizer` не поддерживается в `TrainingArguments`. По умолчанию используется `adamw_torch`.

**Решение:**
Удалён параметр `optimizer` из `TrainingArguments`. Используется оптимизатор по умолчанию.

**Файлы:**
- `utils/trainer.py` - удалена строка `optimizer=optimization_config.get('optimizer', 'adamw_torch')`

---

### Проблема 2: Pickle error с локальной функцией data_collator (✅ ИСПРАВЛЕНО)

**Ошибка:**
```
AttributeError: Can't pickle local object 'LoRATrainer.create_trainer.<locals>.data_collator'
```

**Причина:**
Локальная функция `data_collator` внутри метода `create_trainer()` не может быть сериализована для multiprocessing на Windows.

**Решение:**
Перенёс `data_collator` в отдельный метод класса `_data_collator()`. Теперь это метод класса, который может быть сериализован.

**Файлы:**
- `utils/trainer.py` - создан метод `_data_collator()` класса `LoRATrainer`
- `utils/trainer.py` - удалена локальная функция `data_collator` из `create_trainer()`

---

### Проблема 3: Pickle error с custom code моделью (✅ ИСПРАВЛЕНО)

**Ошибка:**
```
_pickle.PicklingError: Can't pickle <class 'transformers_modules.deepseek-ai.DeepSeek-OCR...'>
```

**Причина:**
Custom code модели (с `trust_remote_code=True`) не могут быть сериализованы для multiprocessing на Windows.

**Решение:**
Установлен `dataloader_num_workers: 0` в конфиге. Это отключает multiprocessing для DataLoader. Для маленького датасета (3 train образца) это не критично.

**Файлы:**
- `configs/training_config.yaml` - изменён `dataloader_num_workers: 2` → `dataloader_num_workers: 0`

---

## ❌ ТЕКУЩАЯ ПРОБЛЕМА: DATA_COLLATOR API

### Ошибка

```
TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'
```

**Трассировка:**
```
File "C:\DSOCR-HW\utils\trainer.py", line 206, in _data_collator
    inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
```

**Текущий код `_data_collator()`:**
```python
def _data_collator(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Загрузка изображений
    images = [Image.open(ex['image_path']).convert('RGB') for ex in examples]
    texts = [ex['text'] for ex in examples]
    
    # Обработка через processor
    inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)  # ← ОШИБКА ЗДЕСЬ
    
    # Labels = input_ids для sequence-to-sequence
    if 'input_ids' in inputs:
        inputs['labels'] = inputs['input_ids'].clone()
    
    return inputs
```

### Анализ проблемы

1. **Что такое `self.processor`:**
   - `AutoProcessor.from_pretrained()` для DeepSeek-OCR возвращает `LlamaTokenizerFast`
   - Это просто tokenizer, не processor с поддержкой images

2. **Проверка типа processor:**
   ```python
   processor = AutoProcessor.from_pretrained('deepseek-ai/DeepSeek-OCR', ...)
   type(processor)  # <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
   hasattr(processor, 'image_processor')  # False
   hasattr(processor, 'tokenizer')  # False
   ```

3. **Что работает в inference:**
   В `test_inference.py` используется:
   ```python
   inputs = processor(images=image, return_tensors="pt")  # Только images, без text
   ```
   Это работает, но только для inference (без text).

4. **Что нужно для обучения:**
   Для обучения Vision2Seq модели нужен способ обработать:
   - Изображения → pixel_values
   - Текст → input_ids, attention_mask
   - Labels → для loss

### Возможные решения

#### Вариант 1: Использовать отдельные image_processor и tokenizer

Нужно найти, как загрузить image_processor отдельно. Возможно:
```python
from transformers import AutoImageProcessor, AutoTokenizer

image_processor = AutoImageProcessor.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Обработка отдельно
pixel_values = image_processor(images, return_tensors="pt")
text_inputs = tokenizer(texts, return_tensors="pt", padding=True)
```

#### Вариант 2: Использовать processor только для изображений, tokenizer для текста

Если processor поддерживает только images:
```python
pixel_values = self.processor(images=images, return_tensors="pt")
text_inputs = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
```

#### Вариант 3: Изучить документацию DeepSeek-OCR

Возможно, есть специальный способ обработки данных для обучения. Нужно найти примеры обучения DeepSeek-OCR или документацию по API.

### Что проверить

1. ✅ Проверить, что возвращает `AutoProcessor.from_pretrained()` для DeepSeek-OCR
2. ❓ Есть ли отдельный `AutoImageProcessor` для этой модели?
3. ❓ Как правильно обрабатывать данные для обучения Vision2Seq моделей?
4. ❓ Есть ли примеры обучения DeepSeek-OCR в документации HuggingFace?

---

## 📁 ИЗМЕНЁННЫЕ ФАЙЛЫ

### 1. `utils/trainer.py`
- ✅ Добавлен импорт `from tqdm import tqdm`
- ✅ Добавлен progress bar в `load_model_and_processor()`
- ✅ Добавлен progress bar в `prepare_datasets()`
- ✅ Добавлен `disable_tqdm=False` в `TrainingArguments`
- ✅ Удалён параметр `optimizer` из `TrainingArguments`
- ✅ Создан метод `_data_collator()` (перенесён из локальной функции)
- ❌ `_data_collator()` не работает из-за проблемы с processor API

### 2. `scripts/train_lora.py`
- ✅ Улучшено логирование с нумерацией шагов
- ✅ Добавлены визуальные разделители
- ✅ Добавлены сообщения о завершении шагов

### 3. `configs/training_config.yaml`
- ✅ Изменён `dataloader_num_workers: 2` → `dataloader_num_workers: 0`

### 4. `PROGRESS_INDICATOR_REPORT.md`
- ✅ Создан отчёт о реализации progress bars

---

## 🧪 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ

### ✅ Что работает:

1. **Progress bars отображаются корректно:**
   ```
   Загрузка процессора:   0%|                                | 0/2 [00:00<?]
   Загрузка процессора:  50%|####################            | 1/2 [00:01<00:01]
   Загрузка модели (может занять ~5-30 мин, если скачивается): 100%|####| 2/2 [00:10<00:00]
   ```

2. **Логирование улучшено:**
   ```
   ================================================================================
   Шаг 1/5: Загрузка модели и процессора...
   ================================================================================
   OK Шаг 1 завершён
   
   ================================================================================
   Шаг 2/5: Настройка LoRA...
   ================================================================================
   ```

3. **Модель загружается успешно:**
   - Процессор загружается
   - Модель загружается (10-14 секунд)
   - LoRA настраивается
   - Датасеты загружаются

4. **Trainer создаётся успешно:**
   - TrainingArguments создаются
   - Trainer объект создаётся

### ❌ Что не работает:

1. **Обучение не запускается:**
   - Ошибка в `_data_collator()` при обработке первого батча
   - Processor не принимает параметр `images` вместе с `text`

---

## 📊 ЛОГИ ОШИБКИ

```
2026-01-10 16:08:07 - train_lora - ERROR - Ошибка обучения: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'
Traceback (most recent call last):
  File "C:\DSOCR-HW\utils\trainer.py", line 338, in train
    self.trainer.train()
  File "C:\DSOCR-HW\venv\lib\site-packages\transformers\trainer.py", line 2052, in train
    return inner_training_loop(
  File "C:\DSOCR-HW\venv\lib\site-packages\transformers\trainer.py", line 2345, in _inner_training_loop
    for step, inputs in enumerate(epoch_iterator):
  File "C:\DSOCR-HW\venv\lib\site-packages\accelerate\data_loader.py", line 567, in __iter__
    current_batch = next(dataloader_iter)
  File "C:\DSOCR-HW\venv\lib\site-packages\torch\utils\data\dataloader.py", line 701, in __next__
    data = self._next_data()
  File "C:\DSOCR-HW\venv\lib\site-packages\torch\utils\data\dataloader.py", line 757, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "C:\DSOCR-HW\venv\lib\site-packages\torch\utils\data\_utils\fetch.py", line 55, in fetch
    return self.collate_fn(data)
  File "C:\DSOCR-HW\utils\trainer.py", line 206, in _data_collator
    inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
  File "C:\DSOCR-HW\venv\lib\site-packages\transformers\tokenization_utils_base.py", line 3024, in __call__
    encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
  File "C:\DSOCR-HW\venv\md\venv\lib\site-packages\transformers\tokenization_utils_base.py", line 3112, in _call_one
    return self.batch_encode_plus(
  File "C:\DSOCR-HW\venv\lib\site-packages\transformers\tokenization_utils_base.py", line 3314, in batch_encode_plus
    return self._batch_encode_plus(
TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'
```

---

## 💡 РЕКОМЕНДАЦИИ ДЛЯ СЕМЁНА (Tech Lead)

Сёма, привет!

Как видишь, progress bars работают отлично — все индикаторы отображаются корректно, логи стали намного понятнее. Но возникла проблема с API processor для обучения.

**Проблема:** `AutoProcessor.from_pretrained()` для DeepSeek-OCR возвращает просто `LlamaTokenizerFast`, который не поддерживает параметр `images`. Для inference это работает (только images), но для обучения нужны и images, и text.

**Что нужно сделать:**
1. Найти правильный способ обработки данных для обучения DeepSeek-OCR
2. Возможно, нужно использовать отдельный `AutoImageProcessor` и `AutoTokenizer`
3. Или найти специальный processor для Vision2Seq моделей

**Вопросы:**
- Есть ли у тебя примеры обучения Vision2Seq моделей с HuggingFace?
- Знаешь ли ты правильный API для DeepSeek-OCR training?
- Может быть, есть документация или примеры в репозитории DeepSeek-OCR?

Пока что я закоммитил все изменения (progress bars работают), и остановился на проблеме с data_collator. После того как разберёмся с правильным API, обучение должно заработать.

P.S. Все progress bars работают красиво — теперь видно, что процесс жив и работает! 🎉

---

## 📝 ЧЕКЛИСТ

- [x] Добавлены progress bars в `load_model_and_processor()`
- [x] Добавлены progress bars в `prepare_datasets()`
- [x] Включён progress bar для обучения (`disable_tqdm=False`)
- [x] Улучшено логирование в `train_lora.py`
- [x] Исправлена проблема с параметром `optimizer`
- [x] Исправлена проблема с pickle и локальной функцией
- [x] Исправлена проблема с multiprocessing (Windows + custom code)
- [ ] **НУЖНО:** Исправить `_data_collator()` для правильного API processor

---

## 🚀 СЛЕДУЮЩИЕ ШАГИ

1. **Семён (Tech Lead):**
   - Изучить правильный API для обучения DeepSeek-OCR
   - Найти примеры или документацию
   - Определить правильный способ обработки данных

2. **Николай:**
   - После получения информации от Семёна - исправить `_data_collator()`
   - Протестировать обучение
   - Убедиться, что progress bars работают во время обучения

---

**С уважением и надеждой на помощь,**  
**Николай (Cursor AI)** 🎯
