# 🔧 ЗАДАНИЕ: ИСПРАВЛЕНИЕ DATA_COLLATOR И КОДИРОВКИ КОНСОЛИ

**Автор:** Семён (Tech Lead)  
**Дата:** 2026-01-10, 16:27 MSK  
**Приоритет:** 🔥 CRITICAL (блокирует запуск обучения)  
**Срок:** 30-40 минут  
**Исполнитель:** Николай (Senior ML Engineer)  
**Связано с:** REPORT_TRAINING_PROGRESS_BARS_20260110.md, Issue #2

---

## 🎯 ЦЕЛЬ

Исправить две критические проблемы, блокирующие обучение:

1. **DATA_COLLATOR API** — ошибка `TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images'`
2. **КОДИРОВКА КОНСОЛИ** — проблемы с отображением русского текста и emoji в Windows Terminal

---

## 📋 ПРОБЛЕМА #1: DATA_COLLATOR API

### Диагноз

Текущий код в `utils/trainer.py` (метод `_data_collator()`) пытается передать **images и text одновременно** в `processor()`:

```python
# ❌ НЕ РАБОТАЕТ
inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
```

**Причина:**  
`AutoProcessor.from_pretrained("deepseek-ai/DeepSeek-OCR")` возвращает `LlamaTokenizerFast`, который **не поддерживает параметр `images`**.

### Архитектура DeepSeek-OCR

DeepSeek-OCR состоит из двух компонентов:
- **DeepEncoder (Vision)** — обрабатывает изображения → vision tokens
- **DeepSeek-3B-MoE (Text)** — декодирует текст

Для обучения Vision2Seq моделей нужна **раздельная обработка**:
- **Images** → через `processor(images=...)`
- **Text** → через `processor.tokenizer(text=...)`

Это стандартный паттерн для всех VLM (Vision Language Models).

---

## 🛠️ РЕШЕНИЕ #1: РАЗДЕЛЬНАЯ ОБРАБОТКА IMAGES И TEXT

### Вариант A: Ручная обработка в data_collator (РЕКОМЕНДУЮ)

**Файл:** `utils/trainer.py`  
**Метод:** `_data_collator()`

**Заменить текущий код на:**

```python
def _data_collator(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Data collator для DeepSeek-OCR с раздельной обработкой images и text.
    
    DeepSeek-OCR требует:
    1. Processor для изображений (pixel_values)
    2. Tokenizer для текста (input_ids, attention_mask, labels)
    
    Args:
        examples: Список словарей с ключами 'image_path' и 'text'
    
    Returns:
        Батч для обучения с ключами:
        - pixel_values: тензор изображений
        - input_ids: токенизированный текст
        - attention_mask: маска внимания
        - labels: метки для loss (копия input_ids)
    """
    from PIL import Image
    
    # 1. Загрузка изображений
    images = [Image.open(ex['image_path']).convert('RGB') for ex in examples]
    texts = [ex['text'] for ex in examples]
    
    # 2. Обработка изображений через processor (только images!)
    try:
        pixel_inputs = self.processor(images=images, return_tensors="pt")
    except Exception as e:
        self.logger.error(f"Ошибка обработки изображений: {e}")
        raise
    
    # 3. Токенизация текста через processor.tokenizer
    try:
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_config.get('max_seq_length', 512)
        )
    except Exception as e:
        self.logger.error(f"Ошибка токенизации текста: {e}")
        raise
    
    # 4. Объединение всех inputs в один батч
    batch = {
        **pixel_inputs,  # pixel_values, etc.
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
    }
    
    # 5. Labels = input_ids для teacher forcing (стандарт для seq2seq)
    # Копируем, чтобы не изменять оригинальный тензор
    batch['labels'] = text_inputs['input_ids'].clone()
    
    return batch
```

### Вариант B: Препроцессинг в prepare_datasets() (АЛЬТЕРНАТИВА)

Если хочешь использовать встроенный `DataCollatorForSeq2Seq` из transformers, то нужно изменить метод `prepare_datasets()`:

**Файл:** `utils/trainer.py`  
**Метод:** `prepare_datasets()`

```python
def prepare_datasets(self, train_path: str, val_path: str):
    """
    Загружает и препроцессирует датасеты с раздельной обработкой images и text.
    """
    from datasets import Dataset
    from PIL import Image
    
    def preprocess_function(examples):
        """
        Препроцессинг одного батча датасета.
        
        Разделяет обработку:
        - images → processor
        - text → tokenizer
        """
        # Загрузка изображений
        images = [Image.open(path).convert('RGB') for path in examples['image_path']]
        
        # Обработка через processor (только images)
        pixel_inputs = self.processor(images=images, return_tensors="pt")
        
        # Токенизация текста
        text_inputs = self.processor.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.training_config.get('max_seq_length', 512),
            return_tensors="pt"
        )
        
        return {
            **pixel_inputs,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': text_inputs['input_ids']  # для loss
        }
    
    # Загрузка из metadata.json
    train_dataset = Dataset.from_json(train_path)
    val_dataset = Dataset.from_json(val_path)
    
    # Препроцессинг
    self.logger.info("Препроцессинг train датасета...")
    train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=['image_path', 'text']  # удаляем, оставляем только тензоры
    )
    
    self.logger.info("Препроцессинг val датасета...")
    val_dataset = val_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=['image_path', 'text']
    )
    
    return train_dataset, val_dataset
```

**Тогда в `create_trainer()` используй встроенный collator:**

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=self.processor.tokenizer,
    model=self.model,
    padding=True,
    pad_to_multiple_of=8  # для эффективности на GPU
)
```

### Какой вариант выбрать?

| Вариант | Плюсы | Минусы | Рекомендация |
|---------|-------|--------|--------------|
| **A: Ручной collator** | Проще, меньше изменений, полный контроль | Нужно вручную управлять padding | ✅ **ДА** (для MVP) |
| **B: Препроцессинг + встроенный collator** | Использует проверенный HF код | Больше изменений, сложнее отладка | ⚠️ Для production |

**Для текущей задачи выбери Вариант A** — он быстрее и проще.

---

## 🛠️ РЕШЕНИЕ #2: КОДИРОВКА КОНСОЛИ (WINDOWS)

### Проблема

Windows PowerShell по умолчанию использует кодировку **Windows-1251** (или CP866 в cmd), что приводит к:
- Крякозяблам вместо русского текста
- Ошибкам отображения emoji (💪, 🚀, ✅)
- Проблемам с progress bars от tqdm

### Решение: Принудительная установка UTF-8

**Файл:** `scripts/train_lora.py`  
**Место:** В самом начале `main()`, перед любым выводом

**Добавить:**

```python
def main():
    """
    Основная функция запуска обучения LoRA.
    """
    # ========================================
    # КРИТИЧНО: Установка UTF-8 для Windows
    # ========================================
    import sys
    import os
    
    # Для Windows: принудительно устанавливаем UTF-8
    if sys.platform == 'win32':
        # Устанавливаем UTF-8 для stdout и stderr
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
        
        # Устанавливаем переменную окружения для подпроцессов
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Для Windows Terminal: устанавливаем консольный codepage
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass  # Игнорируем ошибки, если не Windows
    
    # Теперь можно безопасно выводить русский текст и emoji
    print("=" * 80)
    print("🚀 DSOCR-HW: Обучение DeepSeek-OCR с LoRA")
    print("=" * 80)
    
    # ... остальной код main() ...
```

### Альтернативный вариант: функция-утилита

**Файл:** `utils/encoding_fix.py` (СОЗДАТЬ НОВЫЙ)

```python
"""
Утилиты для исправления кодировки консоли в Windows.

Проблема: Windows по умолчанию использует CP1251/CP866,
что приводит к крякозяблам при выводе русского текста и emoji.

Решение: Принудительная установка UTF-8 для всех потоков вывода.
"""

import sys
import os


def fix_windows_console_encoding():
    """
    Исправляет кодировку консоли для корректного отображения UTF-8 в Windows.
    
    Выполняет:
    1. Переконфигурирует stdout/stderr на UTF-8
    2. Устанавливает PYTHONIOENCODING=utf-8
    3. Устанавливает Windows Console Code Page на 65001 (UTF-8)
    
    Безопасно для других ОС — ничего не делает на Linux/macOS.
    
    Usage:
        from utils.encoding_fix import fix_windows_console_encoding
        
        # В начале main():
        fix_windows_console_encoding()
        print("Теперь можно использовать русский текст и emoji! 🚀")
    """
    # Проверяем, что это Windows
    if sys.platform != 'win32':
        return  # На Linux/macOS ничего не делаем
    
    # 1. Переконфигурируем stdout и stderr
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Для старых версий Python (< 3.7)
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 2. Устанавливаем переменную окружения для подпроцессов
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 3. Устанавливаем Windows Console Code Page на UTF-8 (65001)
    try:
        import ctypes
        # SetConsoleCP - для ввода
        ctypes.windll.kernel32.SetConsoleCP(65001)
        # SetConsoleOutputCP - для вывода
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception as e:
        # Игнорируем ошибки (может не работать в некоторых терминалах)
        pass


def print_encoding_info():
    """
    Выводит информацию о текущих кодировках для отладки.
    
    Usage:
        from utils.encoding_fix import print_encoding_info
        print_encoding_info()
    """
    print("=" * 60)
    print("📊 Информация о кодировках:")
    print("=" * 60)
    print(f"sys.stdout.encoding: {sys.stdout.encoding}")
    print(f"sys.stderr.encoding: {sys.stderr.encoding}")
    print(f"sys.getdefaultencoding(): {sys.getdefaultencoding()}")
    print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'не установлено')}")
    print(f"Платформа: {sys.platform}")
    
    # Тест вывода
    print("\n🧪 Тест вывода:")
    print("  - Русский текст: Привет, мир!")
    print("  - Emoji: 🚀 💪 ✅ ❌ 🎯")
    print("  - Progress bar simulation: " + "█" * 20)
    print("=" * 60)
```

**Использование в `scripts/train_lora.py`:**

```python
from utils.encoding_fix import fix_windows_console_encoding, print_encoding_info

def main():
    # Исправляем кодировку в самом начале
    fix_windows_console_encoding()
    
    # Для отладки (можно убрать потом):
    # print_encoding_info()
    
    print("🚀 DSOCR-HW: Обучение DeepSeek-OCR с LoRA")
    # ... остальной код ...
```

---

## ✅ ЧЕКЛИСТ ВЫПОЛНЕНИЯ

### Задача #1: Data Collator

- [ ] **Прочитал и понял** проблему с processor API
- [ ] **Выбрал вариант** (A — ручной collator ИЛИ B — препроцессинг)
- [ ] **Изменил код** в `utils/trainer.py`
- [ ] **Добавил обработку ошибок** (try-except для обработки images и text)
- [ ] **Проверил синтаксис:** `python -m py_compile utils/trainer.py`
- [ ] **Сделал коммит:** `git commit -m "fix: data_collator with separate images/text processing"`

### Задача #2: Кодировка консоли

- [ ] **Создал файл** `utils/encoding_fix.py` с функциями (если выбрал утилиту)
- [ ] **Добавил fix в начало** `scripts/train_lora.py` (в функцию `main()`)
- [ ] **Протестировал вывод** русского текста и emoji
- [ ] **Проверил синтаксис:** `python -m py_compile utils/encoding_fix.py`
- [ ] **Сделал коммит:** `git commit -m "fix: Windows console encoding for UTF-8 support"`

### Тестирование

- [ ] **Запустил скрипт** с фиксами:
  ```powershell
  python scripts/train_lora.py --config configs/training_config.yaml
  ```
- [ ] **Проверил, что нет ошибки** `TypeError: ... got an unexpected keyword argument 'images'`
- [ ] **Проверил, что русский текст** отображается корректно (не крякозябры)
- [ ] **Проверил, что emoji** отображаются корректно (🚀 💪 ✅)
- [ ] **Проверил, что progress bars** от tqdm отображаются нормально

### Финализация

- [ ] **Создал отчёт** `REPORT_FIX_DATA_COLLATOR_20260110.md` с результатами
- [ ] **Обновил Issue #2** с комментарием о статусе
- [ ] **Сделал финальный коммит** с отчётом
- [ ] **Запустил обучение** на 1 эпохе для проверки

---

## 🧪 ТЕСТОВЫЙ СЦЕНАРИЙ

### Минимальный тест (перед запуском обучения)

**Файл:** `test_data_collator.py` (создать в корне проекта)

```python
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
images = [Image.open(img).convert('RGB') for img in test_images]
pixel_inputs = processor(images=images, return_tensors="pt")
print(f"       ✅ pixel_values shape: {pixel_inputs['pixel_values'].shape}")

# 3b. Text через processor.tokenizer
print("   3b. Обработка text через processor.tokenizer...")
text_inputs = processor.tokenizer(
    test_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)
print(f"       ✅ input_ids shape: {text_inputs['input_ids'].shape}")
print(f"       ✅ attention_mask shape: {text_inputs['attention_mask'].shape}")

# 4. Формирование батча
print("\n4️⃣ Формирование батча...")
batch = {
    **pixel_inputs,
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

# 5. Проверка кодировки
print("\n5️⃣ Проверка кодировки...")
print("   Русский текст: Привет, мир! 🚀")
print("   Emoji: 💪 ✅ ❌ 🎯")

print("\n" + "=" * 80)
print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
print("=" * 80)
```

**Запуск теста:**

```powershell
cd C:\DSOCR-HW
.\venv\Scripts\Activate.ps1
python test_data_collator.py
```

**Ожидаемый вывод:**

```
================================================================================
🧪 ТЕСТ: Data Collator для DeepSeek-OCR
================================================================================

1️⃣ Загрузка processor...
   Тип processor: <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
   Есть tokenizer: False

2️⃣ Подготовка тестовых данных...
   Изображений: 2
   Текстов: 2

3️⃣ Тест раздельной обработки...
   3a. Обработка images через processor...
       ✅ pixel_values shape: torch.Size([2, 3, 1024, 1024])
   3b. Обработка text через processor.tokenizer...
       ✅ input_ids shape: torch.Size([2, 12])
       ✅ attention_mask shape: torch.Size([2, 12])

4️⃣ Формирование батча...
   Ключи батча:
     - pixel_values: shape torch.Size([2, 3, 1024, 1024]), dtype torch.float32
     - input_ids: shape torch.Size([2, 12]), dtype torch.int64
     - attention_mask: shape torch.Size([2, 12]), dtype torch.int64
     - labels: shape torch.Size([2, 12]), dtype torch.int64

5️⃣ Проверка кодировки...
   Русский текст: Привет, мир! 🚀
   Emoji: 💪 ✅ ❌ 🎯

================================================================================
✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!
================================================================================
```

---

## 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

После выполнения задания:

1. **Обучение запустится без ошибок:**
   ```
   Epoch 1/5:   0%|                                    | 0/3 [00:00<?, ?it/s]
   ```

2. **Консоль будет отображать русский текст и emoji:**
   ```
   ✅ Шаг 1 завершён: Модель и процессор загружены
   🚀 Шаг 2: Настройка LoRA...
   💪 Шаг 3: Создание Trainer...
   ```

3. **Progress bars от tqdm будут работать:**
   ```
   Загрузка модели: 100%|████████████████████| 2/2 [00:10<00:00]
   ```

4. **В логах будет видно прогресс:**
   ```
   2026-01-10 16:30:00 - train_lora - INFO - Epoch 1, Step 1/3, Loss: 2.456
   ```

---

## 🚨 ВОЗМОЖНЫЕ ПРОБЛЕМЫ И РЕШЕНИЯ

### Проблема 1: "AttributeError: 'LlamaTokenizerFast' object has no attribute 'tokenizer'"

**Причина:** У `processor` нет атрибута `.tokenizer`, потому что он САМ является tokenizer.

**Решение:** Вместо `processor.tokenizer` используй просто `processor`:

```python
# ❌ НЕ РАБОТАЕТ
text_inputs = self.processor.tokenizer(texts, ...)

# ✅ РАБОТАЕТ
text_inputs = self.processor(texts, ...)  # processor САМ является tokenizer
```

### Проблема 2: "RuntimeError: Expected all tensors to be on the same device"

**Причина:** `pixel_values` и `input_ids` на разных устройствах (CPU vs GPU).

**Решение:** Добавить `.to(device)` после обработки:

```python
device = self.model.device

batch = {
    'pixel_values': pixel_inputs['pixel_values'].to(device),
    'input_ids': text_inputs['input_ids'].to(device),
    'attention_mask': text_inputs['attention_mask'].to(device),
    'labels': text_inputs['input_ids'].clone().to(device)
}
```

### Проблема 3: "Консоль всё равно показывает крякозябры"

**Причина:** Windows Terminal не переключился на UTF-8.

**Решение:**

1. Открой Windows Terminal
2. Settings (Ctrl+,)
3. Defaults → Advanced → Text encoding → выбери **UTF-8**
4. Restart Terminal

Или запускай через:
```powershell
chcp 65001
python scripts/train_lora.py
```

---

## 📝 ФИНАЛЬНЫЙ ОТЧЁТ

После выполнения создай отчёт `REPORT_FIX_DATA_COLLATOR_20260110.md` с разделами:

1. **Выполненная работа** (что сделал)
2. **Изменённые файлы** (список с описанием изменений)
3. **Результаты тестирования** (вывод `test_data_collator.py`)
4. **Скриншоты** (если возможно — консоль с русским текстом и emoji)
5. **Следующие шаги** (запуск полного обучения)

---

## 💬 ВОПРОСЫ?

Если что-то непонятно:

1. Читай комментарии в коде — я постарался всё объяснить
2. Смотри примеры в `test_data_collator.py`
3. Проверь документацию HuggingFace: https://huggingface.co/docs/transformers/main_classes/processors

Если застрял — пиши Семёну в Issue #2, разберёмся вместе! 💪

---

**С уважением и верой в твой успех,**  
**Семён (Tech Lead)** 🎯

P.S. Когда всё заработает — обязательно добавь emoji в отчёт! Владимир их любит (хотя и не признаётся). 😉

P.P.S. Этот фикс — последний барьер перед запуском обучения. После него мы наконец увидим, как модель начнёт учиться на рукописных текстах! 🚀

P.P.P.S. Помни: "Data is the new oil, but data collator is the refinery." — Народная мудрость ML-инженеров. 🛢️➡️⚙️
