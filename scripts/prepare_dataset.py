#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подготовка датасета для fine-tuning DeepSeek-OCR

Автор: Николай
Дата: 2026-01-08
Проект: DSOCR-HW
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict, Any

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
        self.raw_images_dir = Path(raw_images_dir)
        self.annotations_path = Path(annotations_path)
        self.output_dir = Path(output_dir)
        
        # Настройка логгера
        log_file = Path("logs") / f"prepare_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger("prepare_dataset", log_file)
        
        # Инициализация компонентов
        self.loader = DatasetLoader(self.annotations_path, self.raw_images_dir, logger=self.logger)
        self.validator = DataValidator(logger=self.logger)
        self.processor = ImageProcessor(max_size=max_image_size, logger=self.logger)
        self.splitter = DatasetSplitter(train_ratio=train_ratio, random_seed=random_seed, logger=self.logger)
    
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
            self._process_and_save(train_samples, self.output_dir / "train", "train")
            
            # 5. Обработка и сохранение val
            self.logger.info("Шаг 5: Обработка val...")
            self._process_and_save(val_samples, self.output_dir / "val", "val")
            
            self.logger.info("=" * 80)
            self.logger.info("Подготовка датасета завершена успешно!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке датасета: {e}", exc_info=True)
            raise
    
    def _process_and_save(self, samples: List[Dict[str, Any]], output_dir: Path, split_name: str) -> None:
        """Обработка и сохранение образцов.
        
        Args:
            samples: Список образцов
            output_dir: Директория для сохранения (train или val)
            split_name: Имя split (train или val)
        """
        try:
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
            self.splitter.save_metadata(samples, metadata_path, split_name)
            
            self.logger.info(f"Сохранено в {output_dir}: {len(samples)} образцов")
            
        except Exception as e:
            self.logger.error(f"Ошибка _process_and_save: {e}", exc_info=True)
            raise


def main(args):
    """Главная функция.
    
    Args:
        args: Аргументы командной строки
    
    Returns:
        Exit code (0 = успех, 1 = ошибка)
    """
    try:
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
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


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
