# Issue #1: Реализация prepare_dataset.py

**Ответственный**: Николай (Cursor AI)  
**Дата создания**: 08.01.2026  
**Приоритет**: High  
**Статус**: Open  
**Теги**: `data-processing`, `oop`, `module`, `issue-001`

---

## 🎯 Цель задачи

Создать модульный, переиспользуемый скрипт `prepare_dataset.py` с использованием **ООП-архитектуры** для подготовки датасета рукописных текстов к fine-tuning DeepSeek-OCR.

**Ключевые требования**:
- ✅ **Модульность**: все компоненты (загрузка, валидация, предобработка, split) — отдельные классы
- ✅ **Переиспользуемость**: классы должны работать в любом проекте с минимальными изменениями
- ✅ **ООП**: инкапсуляция, наследование, полиморфизм
- ✅ **Type hints**: для всех методов и атрибутов
- ✅ **Docstrings**: Google style для всех классов и методов
- ✅ **Логирование**: через `utils/logger.py`
- ✅ **Обработка ошибок**: try-except с логированием traceback

---

## 📦 Архитектура решения

### Модульная структура

```
utils/
├── data_loader.py           # Класс DatasetLoader
├── data_validator.py        # Класс DataValidator
├── image_processor.py       # Класс ImageProcessor (уже есть, расширить)
├── dataset_splitter.py      # Класс DatasetSplitter
└── logger.py                # Настройка логирования (уже есть)

scripts/
└── prepare_dataset.py       # Главный скрипт (оркестрирует классы)
```

### Принципы проектирования

1. **Single Responsibility Principle (SRP)**: каждый класс отвечает за одну задачу
2. **Open/Closed Principle (OCP)**: классы открыты для расширения, закрыты для модификации
3. **Dependency Injection**: конфигурация передаётся через конструктор
4. **Strategy Pattern**: разные стратегии предобработки изображений

---

## 📋 Технические требования

### Входные данные

**1. Сырые изображения**: `data/raw/`
- Форматы: JPG, PNG, TIFF
- Любое разрешение
- RGB или Grayscale

**2. Аннотации**: `data/annotated/annotations.csv`

**Формат CSV** (UTF-8, с кавычками для многострочного текста):
```csv
filename,text
handwriting_001.jpg,"Вопрос о том - сможем ли мы реализовать
систему дообучения рукописному тексту
в настоящий момент не имеет ответа."
handwriting_002.png,"Другой пример текста."
```

**Обязательные поля**:
- `filename` (str): имя файла из `data/raw/`
- `text` (str): ground truth транскрипция (может быть многострочной)

**Опциональные поля** (для будущего расширения):
- `language` (str): код языка (ru, en, ru-en)
- `quality` (str): качество скана (excellent, good, medium, poor)
- `notes` (str): комментарии

### Выходные данные

**Структура**:
```
data/processed/
├── train/
│   ├── images/                 # Предобработанные изображения
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   └── metadata.json          # Аннотации для train
└── val/
    ├── images/
    │   ├── 00001.jpg
    │   └── ...
    └── metadata.json          # Аннотации для val
```

**Формат metadata.json**:
```json
[
  {
    "image_id": "00001",
    "image_path": "images/00001.jpg",
    "text": "Текст с изображения\nМногострочный",
    "original_filename": "handwriting_001.jpg",
    "width": 1024,
    "height": 768,
    "split": "train"
  },
  ...
]
```

### Split параметры

- **Train**: 80% образцов
- **Val**: 20% образцов
- **Стратификация**: по длине текста (короткие/средние/длинные) — опционально
- **Random seed**: 42 (для воспроизводимости)

### Предобработка изображений

**Базовая** (обязательная):
1. **Resize**: max dimension = 1024px (сохранить aspect ratio)
2. **Normalize**: RGB → [0, 1]
3. **Format**: сохранить как JPG (quality=95)

**Расширенная** (опционально, для будущего):
4. **Augmentation** (только для train):
   - Поворот: ±5°
   - Яркость: ±10%
   - Контраст: ±10%
5. **Grayscale** → RGB (если исходник grayscale)
6. **Binarization** (для улучшения контраста)

---

## 🏗️ Детальная спецификация классов

### 1. Класс `DatasetLoader` (utils/data_loader.py)

**Ответственность**: Загрузка CSV-аннотаций и изображений

```python
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image


class DatasetLoader:
    """Загрузчик датасета из CSV и изображений.
    
    Attributes:
        annotations_path: Путь к CSV-файлу с аннотациями
        images_dir: Директория с сырыми изображениями
    """
    
    def __init__(self, annotations_path: Path, images_dir: Path):
        """Инициализация загрузчика.
        
        Args:
            annotations_path: Путь к annotations.csv
            images_dir: Путь к data/raw/
        
        Raises:
            FileNotFoundError: Если файлы не найдены
        """
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Проверка существования путей."""
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
    
    def load_annotations(self) -> pd.DataFrame:
        """Загрузка аннотаций из CSV.
        
        Returns:
            DataFrame с колонками: filename, text, [language, quality, notes]
        
        Raises:
            ValueError: Если CSV имеет неправильный формат
        """
        try:
            df = pd.read_csv(self.annotations_path, encoding='utf-8')
            
            # Проверка обязательных колонок
            required_columns = ['filename', 'text']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Удаление строк с пустыми значениями
            df = df.dropna(subset=required_columns)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load annotations: {e}")
    
    def load_image(self, filename: str) -> Image.Image:
        """Загрузка одного изображения.
        
        Args:
            filename: Имя файла (например, 'handwriting_001.jpg')
        
        Returns:
            PIL Image объект
        
        Raises:
            FileNotFoundError: Если изображение не найдено
        """
        image_path = self.images_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return Image.open(image_path).convert('RGB')
    
    def load_dataset(self) -> List[Dict[str, any]]:
        """Загрузка полного датасета (аннотации + изображения).
        
        Returns:
            Список словарей с ключами:
                - 'filename': str
                - 'text': str
                - 'image': PIL.Image
                - 'width': int
                - 'height': int
        
        Raises:
            FileNotFoundError: Если изображения отсутствуют
        """
        annotations = self.load_annotations()
        dataset = []
        
        for idx, row in annotations.iterrows():
            filename = row['filename']
            text = row['text']
            
            try:
                image = self.load_image(filename)
                width, height = image.size
                
                dataset.append({
                    'filename': filename,
                    'text': text,
                    'image': image,
                    'width': width,
                    'height': height
                })
            except FileNotFoundError as e:
                # Логируем и пропускаем отсутствующие изображения
                print(f"Warning: {e}")
                continue
        
        return dataset
```

---

### 2. Класс `DataValidator` (utils/data_validator.py)

**Ответственность**: Валидация загруженных данных

```python
from typing import List, Dict
import re


class DataValidator:
    """Валидатор датасета.
    
    Проверяет:
    - Наличие обязательных полей
    - Корректность форматов
    - Минимальное качество изображений
    - Корректность текстовых аннотаций
    """
    
    def __init__(self, min_image_size: int = 100, max_text_length: int = 10000):
        """Инициализация валидатора.
        
        Args:
            min_image_size: Минимальный размер изображения (пиксели)
            max_text_length: Максимальная длина текста (символы)
        """
        self.min_image_size = min_image_size
        self.max_text_length = max_text_length
    
    def validate_sample(self, sample: Dict[str, any]) -> Tuple[bool, str]:
        """Валидация одного образца.
        
        Args:
            sample: Словарь с ключами 'filename', 'text', 'image', 'width', 'height'
        
        Returns:
            Tuple (is_valid, error_message)
        """
        # Проверка обязательных полей
        required_fields = ['filename', 'text', 'image', 'width', 'height']
        for field in required_fields:
            if field not in sample:
                return False, f"Missing field: {field}"
        
        # Проверка размера изображения
        if sample['width'] < self.min_image_size or sample['height'] < self.min_image_size:
            return False, f"Image too small: {sample['width']}x{sample['height']}"
        
        # Проверка текста
        text = sample['text'].strip()
        if len(text) == 0:
            return False, "Empty text"
        if len(text) > self.max_text_length:
            return False, f"Text too long: {len(text)} chars"
        
        return True, ""
    
    def validate_dataset(self, dataset: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Валидация всего датасета.
        
        Args:
            dataset: Список образцов
        
        Returns:
            Отфильтрованный список валидных образцов
        
        Raises:
            ValueError: Если датасет пустой после валидации
        """
        valid_samples = []
        invalid_count = 0
        
        for sample in dataset:
            is_valid, error = self.validate_sample(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                print(f"Warning: Invalid sample {sample['filename']}: {error}")
                invalid_count += 1
        
        if len(valid_samples) == 0:
            raise ValueError("No valid samples found in dataset")
        
        print(f"Validation complete: {len(valid_samples)} valid, {invalid_count} invalid")
        return valid_samples
```

---

### 3. Класс `ImageProcessor` (utils/image_processor.py)

**Ответственность**: Предобработка изображений

```python
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np


class ImageProcessor:
    """Процессор изображений с различными стратегиями предобработки.
    
    Attributes:
        max_size: Максимальный размер по большей стороне
        jpeg_quality: Качество JPEG при сохранении
    """
    
    def __init__(self, max_size: int = 1024, jpeg_quality: int = 95):
        """Инициализация процессора.
        
        Args:
            max_size: Максимальный размер изображения (пиксели)
            jpeg_quality: Качество JPEG (1-100)
        """
        self.max_size = max_size
        self.jpeg_quality = jpeg_quality
    
    def resize_keep_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """Resize с сохранением aspect ratio.
        
        Args:
            image: Исходное изображение
        
        Returns:
            Resized изображение
        """
        width, height = image.size
        
        # Если изображение меньше max_size, не меняем
        if max(width, height) <= self.max_size:
            return image
        
        # Вычисляем новый размер
        if width > height:
            new_width = self.max_size
            new_height = int(height * (self.max_size / width))
        else:
            new_height = self.max_size
            new_width = int(width * (self.max_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def normalize(self, image: Image.Image) -> np.ndarray:
        """Нормализация в [0, 1].
        
        Args:
            image: PIL Image
        
        Returns:
            Numpy array (H, W, 3) с значениями [0, 1]
        """
        return np.array(image).astype(np.float32) / 255.0
    
    def process(self, image: Image.Image) -> Image.Image:
        """Полная предобработка изображения.
        
        Args:
            image: Исходное изображение
        
        Returns:
            Обработанное изображение
        """
        # 1. Конвертация в RGB (если grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 2. Resize
        image = self.resize_keep_aspect_ratio(image)
        
        return image
    
    def save(self, image: Image.Image, output_path: Path) -> None:
        """Сохранение изображения.
        
        Args:
            image: PIL Image
            output_path: Путь для сохранения
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, 'JPEG', quality=self.jpeg_quality)
```

---

### 4. Класс `DatasetSplitter` (utils/dataset_splitter.py)

**Ответственность**: Split датасета на train/val

```python
from typing import List, Dict, Tuple
import random
import json
from pathlib import Path


class DatasetSplitter:
    """Разделение датасета на train/val с сохранением.
    
    Attributes:
        train_ratio: Доля train (0.0 - 1.0)
        random_seed: Seed для воспроизводимости
    """
    
    def __init__(self, train_ratio: float = 0.8, random_seed: int = 42):
        """Инициализация splitter.
        
        Args:
            train_ratio: Доля train (например, 0.8 = 80%)
            random_seed: Random seed
        
        Raises:
            ValueError: Если train_ratio не в [0, 1]
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def split(self, dataset: List[Dict[str, any]]) -> Tuple[List[Dict], List[Dict]]:
        """Разделение датасета на train/val.
        
        Args:
            dataset: Список образцов
        
        Returns:
            Tuple (train_samples, val_samples)
        
        Raises:
            ValueError: Если датасет слишком маленький
        """
        if len(dataset) < 2:
            raise ValueError(f"Dataset too small for split: {len(dataset)} samples")
        
        # Shuffle
        shuffled = dataset.copy()
        random.shuffle(shuffled)
        
        # Split
        train_size = int(len(shuffled) * self.train_ratio)
        train_samples = shuffled[:train_size]
        val_samples = shuffled[train_size:]
        
        # Проверка: хотя бы 1 образец в val
        if len(val_samples) == 0:
            val_samples = [train_samples.pop()]
        
        return train_samples, val_samples
    
    def save_metadata(self, samples: List[Dict[str, any]], output_path: Path) -> None:
        """Сохранение metadata.json.
        
        Args:
            samples: Список образцов
            output_path: Путь к metadata.json
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        for idx, sample in enumerate(samples):
            metadata.append({
                'image_id': f"{idx+1:05d}",
                'image_path': f"images/{idx+1:05d}.jpg",
                'text': sample['text'],
                'original_filename': sample['filename'],
                'width': sample['width'],
                'height': sample['height']
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
```

---

### 5. Главный скрипт `prepare_dataset.py` (scripts/prepare_dataset.py)

**Ответственность**: Оркестрирует все классы

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подготовка датасета для fine-tuning DeepSeek-OCR

Автор: Николай
Дата: 08.01.2026
Проект: DSOCR-HW
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger
from utils.data_loader import DatasetLoader
from utils.data_validator import DataValidator
from utils.image_processor import ImageProcessor
from utils.dataset_splitter import DatasetSplitter


class DatasetPreparer:
    """Главный класс для подготовки датасета.
    
    Оркестрирует:
    - DatasetLoader
    - DataValidator
    - ImageProcessor
    - DatasetSplitter
    """
    
    def __init__(self, 
                 raw_images_dir: Path,
                 annotations_path: Path,
                 output_dir: Path,
                 train_ratio: float = 0.8,
                 max_image_size: int = 1024,
                 random_seed: int = 42):
        """Инициализация preparer.
        
        Args:
            raw_images_dir: Путь к data/raw/
            annotations_path: Путь к annotations.csv
            output_dir: Путь к data/processed/
            train_ratio: Доля train
            max_image_size: Максимальный размер изображения
            random_seed: Random seed
        """
        self.raw_images_dir = raw_images_dir
        self.annotations_path = annotations_path
        self.output_dir = output_dir
        
        # Инициализация компонентов
        self.loader = DatasetLoader(annotations_path, raw_images_dir)
        self.validator = DataValidator()
        self.processor = ImageProcessor(max_size=max_image_size)
        self.splitter = DatasetSplitter(train_ratio=train_ratio, random_seed=random_seed)
        
        # Настройка логгера
        log_file = Path("logs") / f"prepare_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger("prepare_dataset", log_file)
    
    def prepare(self) -> None:
        """Главный метод подготовки датасета."""
        try:
            self.logger.info("=" * 80)
            self.logger.info("Начало подготовки датасета")
            self.logger.info("=" * 80)
            
            # 1. Загрузка
            self.logger.info("Шаг 1: Загрузка датасета...")
            dataset = self.loader.load_dataset()
            self.logger.info(f"Загружено образцов: {len(dataset)}")
            
            # 2. Валидация
            self.logger.info("Шаг 2: Валидация датасета...")
            dataset = self.validator.validate_dataset(dataset)
            self.logger.info(f"Валидных образцов: {len(dataset)}")
            
            # 3. Split
            self.logger.info("Шаг 3: Разделение на train/val...")
            train_samples, val_samples = self.splitter.split(dataset)
            self.logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
            
            # 4. Обработка и сохранение train
            self.logger.info("Шаг 4: Обработка train...")
            self._process_and_save(train_samples, self.output_dir / "train")
            
            # 5. Обработка и сохранение val
            self.logger.info("Шаг 5: Обработка val...")
            self._process_and_save(val_samples, self.output_dir / "val")
            
            self.logger.info("=" * 80)
            self.logger.info("Подготовка датасета завершена успешно!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке датасета: {e}", exc_info=True)
            raise
    
    def _process_and_save(self, samples: List[Dict[str, any]], output_dir: Path) -> None:
        """Обработка и сохранение образцов.
        
        Args:
            samples: Список образцов
            output_dir: Директория для сохранения (train или val)
        """
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Обработка изображений
        for idx, sample in enumerate(samples):
            # Предобработка
            processed_image = self.processor.process(sample['image'])
            
            # Сохранение
            output_path = images_dir / f"{idx+1:05d}.jpg"
            self.processor.save(processed_image, output_path)
            
            # Обновление метаданных
            sample['processed_width'], sample['processed_height'] = processed_image.size
        
        # Сохранение metadata.json
        metadata_path = output_dir / "metadata.json"
        self.splitter.save_metadata(samples, metadata_path)
        
        self.logger.info(f"Сохранено в {output_dir}: {len(samples)} образцов")


def main(args):
    """Главная функция."""
    preparer = DatasetPreparer(
        raw_images_dir=Path(args.raw_images),
        annotations_path=Path(args.annotations),
        output_dir=Path(args.output),
        train_ratio=args.train_ratio,
        max_image_size=args.max_size,
        random_seed=args.seed
    )
    
    preparer.prepare()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка датасета для fine-tuning")
    parser.add_argument("--raw-images", type=str, default="data/raw",
                        help="Путь к сырым изображениям")
    parser.add_argument("--annotations", type=str, default="data/annotated/annotations.csv",
                        help="Путь к CSV с аннотациями")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Путь для сохранения обработанных данных")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Доля train (0.0 - 1.0)")
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Максимальный размер изображения")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    sys.exit(main(args))
```

---

## 🧪 Тестирование

### Минимальный тест (2 образца)

**Подготовка**:
1. Положить 2 скана в `data/raw/`: `test_001.jpg`, `test_002.jpg`
2. Создать `data/annotated/annotations.csv`:
```csv
filename,text
test_001.jpg,"Тестовый текст 1"
test_002.jpg,"Тестовый текст 2"
```

**Запуск**:
```powershell
python scripts/prepare_dataset.py --raw-images data/raw --annotations data/annotated/annotations.csv --output data/processed
```

**Ожидаемый результат**:
```
data/processed/
├── train/
│   ├── images/
│   │   └── 00001.jpg
│   └── metadata.json
└── val/
    ├── images/
    │   └── 00001.jpg
    └── metadata.json
```

### Unit-тесты (опционально, для будущего)

```python
# tests/test_data_loader.py
import pytest
from utils.data_loader import DatasetLoader

def test_load_annotations():
    loader = DatasetLoader(
        annotations_path=Path("data/test/annotations.csv"),
        images_dir=Path("data/test/images")
    )
    df = loader.load_annotations()
    assert len(df) > 0
    assert 'filename' in df.columns
    assert 'text' in df.columns
```

---

## 📝 Чеклист перед завершением Issue

- [ ] Все классы реализованы в отдельных модулях (`utils/*.py`)
- [ ] Все методы имеют type hints
- [ ] Все классы и методы имеют docstrings (Google style)
- [ ] Логирование настроено через `utils/logger.py`
- [ ] Обработка ошибок (try-except) для всех критичных операций
- [ ] Код проверен на синтаксис: `python -m py_compile scripts/prepare_dataset.py`
- [ ] Тестирование на минимальном датасете (2 образца)
- [ ] Результаты сохраняются в `data/processed/train/` и `data/processed/val/`
- [ ] Логи созданы в `logs/prepare_dataset_*.log`
- [ ] Git коммит: `git commit -m "feat: implement prepare_dataset.py with OOP architecture (Issue #1)"`
- [ ] Pull Request создан → ждём ревью от Семёна

---

## 🚀 Следующие шаги после Issue #1

1. **Issue #2**: `train_lora.py` — загрузка обработанного датасета и fine-tuning
2. **Issue #3**: `evaluate.py` — расчёт метрик CER/WER
3. **Issue #4**: `inference.py` — инференс с LoRA-адаптером

---

## 📞 Контакты

**Вопросы по задаче**: создай Issue с меткой `question` и тегом `@Tech Lead`  
**Code review**: Pull Request → @Семён  
**Проблемы с данными**: Issue с тегом `@Product Owner`

---

**Удачи, Николай! Ждём твой Pull Request! 🚀**
