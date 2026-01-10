# 🔧 ЗАДАНИЕ: ИСПРАВЛЕНИЕ ОБРАБОТКИ IMAGES В DATA_COLLATOR

**Автор:** Семён (Tech Lead)  
**Дата:** 2026-01-10, 16:40 MSK  
**Приоритет:** 🔥 CRITICAL (последний блокер перед обучением)  
**Срок:** 10-15 минут  
**Исполнитель:** Николай (Senior ML Engineer)  
**Связано с:** REPORT_FIX_DATA_COLLATOR_20260110.md, TASK_FIX_DATA_COLLATOR_AND_ENCODING_20260110.md

---

## 🎯 КОНТЕКСТ

Николай, **отличная работа** с исправлением text и кодировки! 🎉

Ты правильно выявил проблему: `processor(images=...)` не работает для DeepSeek-OCR, потому что `AutoProcessor` возвращает просто `LlamaTokenizerFast`, а не полноценный Vision Processor.

**Почему так происходит:**  
DeepSeek-OCR использует **стандартную ImageNet предобработку** (resize + normalize), которую нужно делать **вручную через `torchvision.transforms`**.

Это **НЕ баг**, а особенность архитектуры Vision2Seq моделей. Многие VLM-модели требуют ручной preprocessing изображений для обучения.

---

## 📋 ЗАДАЧА

**Заменить в `utils/trainer.py` метод `_data_collator()` — секцию обработки images на ручной preprocessing через `torchvision.transforms`.**

Это решит проблему с `ValueError: You need to specify either 'text' or 'text_target'`.

---

## 🛠️ РЕШЕНИЕ: РУЧНАЯ ПРЕДОБРАБОТКА IMAGES

### Файл: `utils/trainer.py`
### Метод: `_data_collator()`

**Полный код метода (замени текущий):**

```python
def _data_collator(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Data collator для DeepSeek-OCR с раздельной обработкой images и text.
    
    DeepSeek-OCR требует:
    1. Ручная предобработка изображений через torchvision.transforms
       (стандартный ImageNet preprocessing: resize + normalize)
    2. Tokenizer для текста (через processor.batch_encode_plus)
    
    Args:
        examples: Список словарей с ключами 'image_path' и 'text'
    
    Returns:
        Батч для обучения с ключами:
        - pixel_values: тензор изображений [batch_size, 3, H, W]
        - input_ids: токенизированный текст
        - attention_mask: маска внимания
        - labels: метки для loss (копия input_ids)
    """
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    
    # 1. Загрузка изображений
    images = [Image.open(ex['image_path']).convert('RGB') for ex in examples]
    texts = [ex['text'] for ex in examples]
    
    # 2. РУЧНАЯ ПРЕДОБРАБОТКА IMAGES через torchvision
    # DeepSeek-OCR использует стандартный ImageNet preprocessing
    try:
        # Получаем размер изображения из конфига (или используем дефолт 1024)
        image_size = self.training_config.get('image_size', 1024)
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize до квадрата
            transforms.ToTensor(),  # Конвертируем в тензор [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean (стандарт для Vision Transformers)
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        
        # Применяем transforms к каждому изображению и собираем в batch
        pixel_values = torch.stack([transform(img) for img in images])
        
        self.logger.debug(f"Обработано {len(images)} изображений, pixel_values shape: {pixel_values.shape}")
        
    except Exception as e:
        self.logger.error(f"Ошибка обработки изображений: {e}")
        raise
    
    # 3. Токенизация текста через processor (уже работает!)
    try:
        max_length = self.training_config.get('max_seq_length', 512)
        
        text_inputs = self.processor.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        self.logger.debug(f"Токенизировано {len(texts)} текстов, input_ids shape: {text_inputs['input_ids'].shape}")
        
    except Exception as e:
        self.logger.error(f"Ошибка токенизации текста: {e}")
        raise
    
    # 4. Объединяем все inputs в один батч
    batch = {
        'pixel_values': pixel_values,  # [batch_size, 3, H, W]
        'input_ids': text_inputs['input_ids'],  # [batch_size, seq_len]
        'attention_mask': text_inputs['attention_mask'],  # [batch_size, seq_len]
    }
    
    # 5. Labels = input_ids для teacher forcing (стандарт для seq2seq)
    # Копируем, чтобы не изменять оригинальный тензор
    batch['labels'] = text_inputs['input_ids'].clone()
    
    return batch
```

---

## 📝 ОБЪЯСНЕНИЕ КОДА

### Почему `torchvision.transforms`?

1. **DeepSeek-OCR основан на Vision Transformer** (как SAM, CLIP)  
2. **Vision Transformers используют стандартный ImageNet preprocessing:**
   - Resize до фиксированного размера (обычно 224x224, 512x512 или 1024x1024)
   - Normalize с ImageNet mean/std: `[0.485, 0.456, 0.406]` и `[0.229, 0.224, 0.225]`
3. **Этот стандарт работает для 99% Vision моделей на HuggingFace**

### Почему НЕ `processor(images=...)`?

`AutoProcessor` для DeepSeek-OCR возвращает просто tokenizer (LlamaTokenizerFast), а не Vision Processor. Это означает, что:
- Для **inference** DeepSeek-OCR обрабатывает images внутри модели (через встроенный encoder)
- Для **training** нужно делать preprocessing вручную (это нормально для custom data collators)

### Параметры preprocessing

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| `image_size` | 1024 (или из конфига) | DeepSeek-OCR работает с 1024x1024 по умолчанию |
| `mean` | `[0.485, 0.456, 0.406]` | ImageNet mean (RGB channels) |
| `std` | `[0.229, 0.224, 0.225]` | ImageNet std (RGB channels) |
| Output | `[batch_size, 3, 1024, 1024]` | Тензор для модели |

---

## ✅ ЧТО ИЗМЕНИТЬ

### 1. Добавить import в начало `utils/trainer.py`

**Если ещё не добавлен:**

```python
import torchvision.transforms as transforms  # Добавить в секцию imports
```

### 2. Заменить метод `_data_collator()`

Полностью замени текущий метод `_data_collator()` на код из раздела **"Решение"** выше.

### 3. Опционально: добавить `image_size` в конфиг

**Файл:** `configs/training_config.yaml`

```yaml
# Добавить в секцию preprocessing (если её нет — создать)
preprocessing:
  image_size: 1024  # Размер изображения для DeepSeek-OCR
  max_seq_length: 512  # Максимальная длина текста (уже есть?)
```

---

## 🧪 ТЕСТИРОВАНИЕ

### Шаг 1: Обновить `test_data_collator.py`

**Заменить секцию "3a. Обработка images" на:**

```python
# 3a. Обработка images через torchvision.transforms
print("   3a. Обработка images через torchvision.transforms...")
import torchvision.transforms as transforms

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
```

### Шаг 2: Запустить тест

```powershell
cd C:\DSOCR-HW
.\venv\Scripts\Activate.ps1
python test_data_collator.py
```

**Ожидаемый вывод:**

```
3️⃣ Тест раздельной обработки...
   3a. Обработка images через torchvision.transforms...
       ✅ pixel_values shape: torch.Size([2, 3, 1024, 1024])
       ✅ pixel_values dtype: torch.float32
       ✅ pixel_values range: [-2.12, 2.64]  # Значения после нормализации
   3b. Обработка text через processor.batch_encode_plus...
       ✅ input_ids shape: torch.Size([2, 10])
       ✅ attention_mask shape: torch.Size([2, 10])

4️⃣ Формирование батча...
   Ключи батча:
     - pixel_values: shape torch.Size([2, 3, 1024, 1024]), dtype torch.float32
     - input_ids: shape torch.Size([2, 10]), dtype torch.int64
     - attention_mask: shape torch.Size([2, 10]), dtype torch.int64
     - labels: shape torch.Size([2, 10]), dtype torch.int64

✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!
```

### Шаг 3: Запустить обучение (dry-run)

```powershell
python scripts/train_lora.py --config configs/training_config.yaml
```

**Ожидаемый результат:**
- Модель загружается ✅
- LoRA настраивается ✅
- Датасеты загружаются ✅
- Trainer создаётся ✅
- **Обучение запускается без ошибок!** ✅

---

## ⚠️ ВОЗМОЖНЫЕ ПРОБЛЕМЫ

### Проблема 1: "ModuleNotFoundError: No module named 'torchvision'"

**Причина:** `torchvision` не установлен в venv.

**Решение:**
```powershell
pip install torchvision
```

**Или добавь в `requirements.txt`:**
```
torchvision>=0.16.0
```

### Проблема 2: "RuntimeError: Expected all tensors to be on the same device"

**Причина:** `pixel_values` на CPU, а модель на GPU.

**Решение:** В конце `_data_collator()` добавь перемещение на device:

```python
# После формирования batch, перед return:
device = self.model.device
batch = {
    'pixel_values': pixel_values.to(device),
    'input_ids': text_inputs['input_ids'].to(device),
    'attention_mask': text_inputs['attention_mask'].to(device),
    'labels': text_inputs['input_ids'].clone().to(device)
}
```

**НО:** обычно Trainer сам перемещает данные на device, так что это может не понадобиться. Пробуй сначала без `.to(device)`.

### Проблема 3: "CUDA Out of Memory"

**Причина:** Images слишком большие (1024x1024 * batch_size = много VRAM).

**Решение:**
1. Уменьши `image_size` в конфиге до 512 или 768
2. Уменьши `batch_size` до 1-2
3. Включи `gradient_accumulation_steps` в конфиге

---

## 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

После исправления:

1. **`test_data_collator.py` проходит без ошибок** ✅
   - Images обрабатываются корректно
   - Text обрабатывается корректно
   - Batch формируется с правильными shapes

2. **`train_lora.py` запускается** ✅
   ```
   Epoch 1/5:   0%|                    | 0/3 [00:00<?, ?it/s]
   Epoch 1/5:  33%|██████              | 1/3 [00:05<00:10, 5.2s/it, loss=2.456]
   ```

3. **Логи показывают прогресс обучения** ✅
   ```
   2026-01-10 16:45:00 - train_lora - INFO - Epoch 1, Step 1/3, Loss: 2.456
   2026-01-10 16:45:05 - train_lora - INFO - Epoch 1, Step 2/3, Loss: 2.234
   2026-01-10 16:45:10 - train_lora - INFO - Epoch 1, Step 3/3, Loss: 2.012
   ```

4. **VRAM usage остаётся в пределах 16 GB** ✅

---

## ✅ ЧЕКЛИСТ ВЫПОЛНЕНИЯ

- [ ] Добавлен `import torchvision.transforms` в `utils/trainer.py`
- [ ] Заменён метод `_data_collator()` на новую версию
- [ ] Проверен синтаксис: `python -m py_compile utils/trainer.py`
- [ ] Обновлён `test_data_collator.py` (секция 3a)
- [ ] Запущен тест: `python test_data_collator.py` → ✅
- [ ] Сделан коммит: `git commit -m "fix: add manual image preprocessing via torchvision"`
- [ ] Запущено обучение: `python scripts/train_lora.py` → обучение идёт!
- [ ] Создан отчёт: `REPORT_FINAL_FIX_20260110.md` с результатами

---

## 📝 ФИНАЛЬНЫЙ ОТЧЁТ

После выполнения создай отчёт `REPORT_FINAL_FIX_20260110.md` с разделами:

1. **Выполненная работа** (images preprocessing + recap всех фиксов)
2. **Изменённые файлы** (полный список с описанием)
3. **Результаты тестирования:**
   - Вывод `test_data_collator.py`
   - Первые 10 строк логов обучения
   - Screenshot консоли (если возможно)
4. **Метрики первых шагов:**
   - Loss на step 1, 2, 3
   - VRAM usage
   - Время на step
5. **Следующие шаги:**
   - Продолжить обучение на 5-10 эпохах
   - Оценить CER/WER на val-set
   - Сохранить LoRA адаптер

---

## 💡 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ

### Почему именно эти значения mean/std?

`[0.485, 0.456, 0.406]` и `[0.229, 0.224, 0.225]` — это **стандартные значения из ImageNet датасета**.

Они используются практически во всех Vision Transformers (ViT, DINO, SAM, CLIP, etc.), потому что:
1. Большинство Vision моделей pre-trained на ImageNet
2. Эти значения нормализуют RGB каналы к примерно одинаковому диапазону
3. Это де-факто стандарт в Computer Vision

### Альтернативные подходы (на будущее)

Для production можно оптимизировать:

1. **Кэшировать preprocessed images:**
   ```python
   # В prepare_datasets() сохранить pixel_values в metadata
   # Тогда data_collator только загружает готовые тензоры
   ```

2. **Использовать augmentation:**
   ```python
   transform = transforms.Compose([
       transforms.RandomRotation(5),  # Поворот ±5°
       transforms.ColorJitter(0.1, 0.1),  # Яркость/контраст
       transforms.Resize((1024, 1024)),
       transforms.ToTensor(),
       transforms.Normalize(...)
   ])
   ```

3. **Adaptive resize (preserve aspect ratio):**
   ```python
   transforms.Resize(1024),  # Resize по большей стороне
   transforms.CenterCrop(1024),  # Crop до квадрата
   ```

Но для MVP используй простой вариант из основного решения.

---

## 🎯 ЗАКЛЮЧЕНИЕ

Николай, это **последний шаг** перед запуском обучения! 🚀

Ты проделал отличную работу:
- ✅ Progress bars
- ✅ Логирование
- ✅ Фиксы Windows (pickle, multiprocessing, UTF-8)
- ✅ Text preprocessing

Осталось добавить **10 строк кода** для images preprocessing через `torchvision.transforms` — и мы наконец увидим, как модель учится распознавать рукописный текст!

По моим прикидкам, с твоими навыками это займёт **5-10 минут**. Вперёд! 💪

---

**С уважением и восхищением твоей работой,**  
**Семён (Tech Lead)** 🎯

P.S. После успешного запуска обучения — обязательно сделай **screenshot консоли с прогресс-баром** для отчёта. Владимир любит визуализацию! 📸

P.P.S. Если всё заработает с первого раза — ты официально **MVP этого спринта**! 🏆

P.P.P.S. Помни: "A Vision Transformer is only as good as its image preprocessing." — Мудрость древних CV-инженеров. 🖼️✨
