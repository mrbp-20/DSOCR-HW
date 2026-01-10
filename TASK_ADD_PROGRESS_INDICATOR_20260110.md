# 📊 ЗАДАНИЕ ДЛЯ НИКОЛАЯ: ДОБАВЛЕНИЕ ИНДИКАТОРА ПРОГРЕССА

**От:** Семён (Tech Lead)  
**Кому:** Николай (Senior ML Engineer)  
**Дата:** 10.01.2026, 15:50 MSK  
**Тема:** Добавить индикатор прогресса в `train_lora.py` и `utils/trainer.py`  
**Приоритет:** HIGH  
**Срок:** 15-20 минут

---

## 🎯 ПРОБЛЕМА

**Ситуация:**
- При запуске `train_lora.py` процесс может зависать на загрузке модели (6-7 GB с HuggingFace)
- Логи пишут только начало, потом тишина 15-30 минут
- **Непонятно:** процесс идёт или завис?

**Пример лога (сейчас):**
```
2026-01-09 23:52:13 - train_lora - INFO - Начало обучения LoRA
2026-01-09 23:52:14 - train_lora - INFO - Модель загружена: deepseek-ai/DeepSeek-OCR
[ТИШИНА 30 МИНУТ...]
```

**Нужно:**
Видеть, что процесс ЖИВОЙ и работает!

---

## ✅ РЕШЕНИЕ

Добавить **индикатор прогресса** с использованием `tqdm` в ключевых местах:

1. **Загрузка модели** (load_model_and_processor)
2. **Подготовка датасетов** (prepare_datasets)
3. **Обучение** (train)

---

## 📝 ИЗМЕНЕНИЯ В КОДЕ

### 1. Добавить импорт `tqdm` в `utils/trainer.py`

**В начало файла `utils/trainer.py`:**

```python
from tqdm import tqdm
import time
```

---

### 2. Изменить `load_model_and_processor()` в `utils/trainer.py`

**Было:**
```python
def load_model_and_processor(self) -> None:
    """Load model and processor."""
    try:
        self.logger.info("Загрузка модели и процессора...")
        
        model_config = self.config['model']
        base_model = model_config['base_model']
        revision = model_config.get('revision', "9f30c71f441d010e5429c532364a86705536c53a")
        
        # Загрузка процессора
        self.processor = AutoProcessor.from_pretrained(...)
        self.logger.info(f"Процессор загружен")
        
        # Загрузка модели
        self.model = AutoModel.from_pretrained(...)
        self.logger.info(f"Модель загружена")
```

**Стало:**
```python
def load_model_and_processor(self) -> None:
    """Load model and processor with progress indication."""
    try:
        self.logger.info("Загрузка модели и процессора...")
        
        model_config = self.config['model']
        base_model = model_config['base_model']
        revision = model_config.get('revision', "9f30c71f441d010e5429c532364a86705536c53a")
        
        # Прогресс-бар для загрузки модели
        with tqdm(total=2, desc="Загрузка модели", unit="step", ncols=100, 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # Загрузка процессора
            pbar.set_description("Загрузка процессора")
            self.processor = AutoProcessor.from_pretrained(
                base_model,
                revision=revision,
                trust_remote_code=model_config.get('trust_remote_code', True)
            )
            pbar.update(1)
            self.logger.info(f"Процессор загружен: {base_model}")
            
            # Загрузка модели (может занять долго, если скачивается)
            pbar.set_description("Загрузка модели (может занять ~5-30 мин, если скачивается)")
            
            torch_dtype = getattr(torch, model_config.get('torch_dtype', 'float16'))
            self.model = AutoModel.from_pretrained(
                base_model,
                revision=revision,
                torch_dtype=torch_dtype,
                attn_implementation=model_config.get('attn_implementation', 'eager'),
                device_map=model_config.get('device_map', 'auto'),
                trust_remote_code=model_config.get('trust_remote_code', True),
                cache_dir=model_config.get('cache_dir')
            )
            pbar.update(1)
            self.logger.info(f"Модель загружена: {base_model} (revision: {revision})")
            
    except Exception as e:
        self.logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
        raise
```

**Что изменилось:**
- ✅ Добавлен `tqdm` progress bar с 2 шагами
- ✅ Показывает описание текущего шага
- ✅ Предупреждает, что загрузка модели может занять 5-30 мин

---

### 3. Изменить `prepare_datasets()` в `utils/trainer.py`

**Добавить progress bar при загрузке metadata:**

```python
def prepare_datasets(self) -> None:
    """Prepare datasets with progress indication."""
    try:
        self.logger.info("Подготовка датасетов...")
        
        data_config = self.config['data']
        train_path = Path(data_config['train_path'])
        val_path = Path(data_config['val_path'])
        
        with tqdm(total=2, desc="Загрузка данных", unit="split", ncols=100) as pbar:
            # Загрузка train metadata
            pbar.set_description("Загрузка train данных")
            train_metadata = self._load_metadata(train_path / 'metadata.json')
            pbar.update(1)
            
            # Загрузка val metadata
            pbar.set_description("Загрузка val данных")
            val_metadata = self._load_metadata(val_path / 'metadata.json')
            pbar.update(1)
        
        self.logger.info(f"Загружено train: {len(train_metadata)}, val: {len(val_metadata)} образцов")
        
        # Остальной код без изменений...
        train_dict = {
            'image_path': [str(train_path / item['image_path']) for item in train_metadata],
            'text': [item['text'] for item in train_metadata]
        }
        val_dict = {
            'image_path': [str(val_path / item['image_path']) for item in val_metadata],
            'text': [item['text'] for item in val_metadata]
        }
        
        self.train_dataset = Dataset.from_dict(train_dict)
        self.eval_dataset = Dataset.from_dict(val_dict)
        
        self.logger.info("Датасеты подготовлены")
        
    except Exception as e:
        self.logger.error(f"Ошибка подготовки датасетов: {e}", exc_info=True)
        raise
```

---

### 4. Добавить progress bar для обучения

**В `create_trainer()` добавить `disable_tqdm=False` в TrainingArguments:**

```python
training_args = TrainingArguments(
    # ... все остальные параметры ...
    disable_tqdm=False,  # ← ДОБАВИТЬ ЭТУ СТРОКУ!
    # ...
)
```

**Это включит встроенный tqdm в Trainer для обучения!**

---

### 5. Добавить "сердцебиение" в основной скрипт

**В `scripts/train_lora.py` добавить печать "сердцебиения" между шагами:**

```python
def main(args):
    try:
        # ... настройка логгера ...
        
        logger.info("=" * 80)
        logger.info("Запуск обучения LoRA")
        logger.info("=" * 80)
        
        trainer = LoRATrainer(config_path=Path(args.config), logger=logger)
        
        # Шаг 1: Загрузка модели
        logger.info("\n" + "="*80)
        logger.info("Шаг 1/5: Загрузка модели и процессора...")
        logger.info("="*80)
        trainer.load_model_and_processor()
        logger.info("✅ Шаг 1 завершён")
        
        # Шаг 2: LoRA
        logger.info("\n" + "="*80)
        logger.info("Шаг 2/5: Настройка LoRA...")
        logger.info("="*80)
        trainer.setup_lora()
        logger.info("✅ Шаг 2 завершён")
        
        # Шаг 3: Датасеты
        logger.info("\n" + "="*80)
        logger.info("Шаг 3/5: Подготовка датасетов...")
        logger.info("="*80)
        trainer.prepare_datasets()
        logger.info("✅ Шаг 3 завершён")
        
        # Шаг 4: Trainer
        logger.info("\n" + "="*80)
        logger.info("Шаг 4/5: Создание Trainer...")
        logger.info("="*80)
        trainer.create_trainer()
        logger.info("✅ Шаг 4 завершён")
        
        # Шаг 5: Обучение
        logger.info("\n" + "="*80)
        logger.info("Шаг 5/5: ЗАПУСК ОБУЧЕНИЯ!")
        logger.info("="*80)
        trainer.train()
        logger.info("✅ Шаг 5 завершён")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 Обучение завершено успешно!")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"❌ Ошибка: {e}", exc_info=True)
        else:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
        return 1
```

---

## 📊 ОЖИДАЕМЫЙ ВЫВОД ПОСЛЕ ИЗМЕНЕНИЙ

**Вместо:**
```
2026-01-09 23:52:14 - train_lora - INFO - Модель загружена
[ТИШИНА...]
```

**Будет:**
```
================================================================================
Шаг 1/5: Загрузка модели и процессора...
================================================================================
Загрузка модели: |██████████| 2/2 [00:15<00:00]
  ✓ Загрузка процессора (завершено)
  ✓ Загрузка модели (завершено, 15m 23s)
✅ Шаг 1 завершён

================================================================================
Шаг 2/5: Настройка LoRA...
================================================================================
...
```

---

## 📝 ЧЕКЛИСТ ДЛЯ НИКОЛАЯ

- [ ] Добавить `from tqdm import tqdm` в `utils/trainer.py`
- [ ] Изменить `load_model_and_processor()` - добавить progress bar
- [ ] Изменить `prepare_datasets()` - добавить progress bar
- [ ] Добавить `disable_tqdm=False` в `TrainingArguments`
- [ ] Улучшить логирование в `train_lora.py` (нумерация шагов, галочки)
- [ ] Протестировать - запустить `train_lora.py`
- [ ] Закоммитить изменения
- [ ] Создать отчёт `PROGRESS_INDICATOR_REPORT.md`

---

## 🚀 ЗАПУСК И ПРОВЕРКА

**После изменений запусти:**

```powershell
cd C:\DSOCR-HW
.\venv\Scripts\Activate.ps1
python scripts/train_lora.py --config configs/training_config.yaml
```

**Проверь:**
- ✅ Появляются progress bars для загрузки модели
- ✅ Появляются progress bars для датасетов
- ✅ Появляются progress bars во время обучения (epoch/step)
- ✅ Логи остаются читаемыми с чёткой нумерацией шагов

---

## 💬 КОММЕНТАРИИ

**Почему это важно:**

1. **Прозрачность** - видно, что процесс живой
2. **Нет паники** - понятно, что загрузка модели - это нормально
3. **Оценка времени** - tqdm показывает примерное время завершения
4. **Дебаг** - если зависание, сразу видно на каком шаге

**P.S.** Николай, если есть идеи по улучшению вывода (например, цветной текст, emoji) - добавляй! 🎨

---

**НИКОЛАЙ, ЭТО ВАЖНОЕ УЛУЧШЕНИЕ!** 👍

**Время:** 15-20 мин  
**Приоритет:** HIGH

---

**С уважением,**  
**Семён (Tech Lead, который теперь не спит)** 👊🔥
