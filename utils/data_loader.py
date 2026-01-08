#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loader for DSOCR-HW project.

Автор: Николай
Дата: 2026-01-08
Проект: DSOCR-HW
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import logging
from typing import Optional


class DatasetLoader:
    """Загрузчик датасета из CSV и изображений.
    
    Attributes:
        annotations_path: Путь к CSV-файлу с аннотациями
        images_dir: Директория с сырыми изображениями
        logger: Логгер для логирования
    """
    
    def __init__(self, annotations_path: Path, images_dir: Path, logger: Optional[logging.Logger] = None):
        """Инициализация загрузчика.
        
        Args:
            annotations_path: Путь к annotations.csv
            images_dir: Путь к data/raw/
            logger: Логгер (опционально)
        
        Raises:
            FileNotFoundError: Если файлы не найдены
        """
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Проверка существования путей."""
        try:
            if not self.annotations_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
            if not self.images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
            self.logger.info(f"Пути валидны: {self.annotations_path}, {self.images_dir}")
        except Exception as e:
            self.logger.error(f"Ошибка валидации путей: {e}", exc_info=True)
            raise
    
    def load_annotations(self) -> pd.DataFrame:
        """Загрузка аннотаций из CSV.
        
        Returns:
            DataFrame с колонками: filename, text, [language, quality, notes]
        
        Raises:
            ValueError: Если CSV имеет неправильный формат
        """
        try:
            df = pd.read_csv(self.annotations_path, encoding='utf-8')
            self.logger.info(f"Загружено записей из CSV: {len(df)}")
            
            # Проверка обязательных колонок
            required_columns = ['filename', 'text']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Удаление строк с пустыми значениями
            df = df.dropna(subset=required_columns)
            self.logger.info(f"Записей после удаления пустых: {len(df)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки аннотаций: {e}", exc_info=True)
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
        try:
            image_path = self.images_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            self.logger.debug(f"Загружено изображение: {filename}, размер: {image.size}")
            return image
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки изображения {filename}: {e}", exc_info=True)
            raise
    
    def load_dataset(self) -> List[Dict[str, Any]]:
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
        try:
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
                        'text': str(text).strip(),
                        'image': image,
                        'width': width,
                        'height': height
                    })
                except FileNotFoundError as e:
                    # Логируем и пропускаем отсутствующие изображения
                    self.logger.warning(f"Пропущено изображение {filename}: {e}")
                    continue
            
            self.logger.info(f"Загружено образцов: {len(dataset)}/{len(annotations)}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки датасета: {e}", exc_info=True)
            raise
