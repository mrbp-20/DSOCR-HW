#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset splitter for DSOCR-HW project.

Автор: Николай
Дата: 2026-01-08
Проект: DSOCR-HW
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging
from typing import Optional


class DatasetSplitter:
    """Разделение датасета на train/val с сохранением.
    
    Attributes:
        train_ratio: Доля train (0.0 - 1.0)
        random_seed: Seed для воспроизводимости
        logger: Логгер для логирования
    """
    
    def __init__(self, train_ratio: float = 0.8, random_seed: int = 42, logger: Optional[logging.Logger] = None):
        """Инициализация splitter.
        
        Args:
            train_ratio: Доля train (например, 0.8 = 80%)
            random_seed: Random seed
            logger: Логгер (опционально)
        
        Raises:
            ValueError: Если train_ratio не в [0, 1]
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.logger = logger or logging.getLogger(__name__)
        random.seed(random_seed)
    
    def split(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Разделение датасета на train/val.
        
        Args:
            dataset: Список образцов
        
        Returns:
            Tuple (train_samples, val_samples)
        
        Raises:
            ValueError: Если датасет слишком маленький
        """
        try:
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
            
            self.logger.info(f"Split: Train={len(train_samples)}, Val={len(val_samples)}")
            return train_samples, val_samples
            
        except Exception as e:
            self.logger.error(f"Ошибка split: {e}", exc_info=True)
            raise
    
    def save_metadata(self, samples: List[Dict[str, Any]], output_path: Path, split_name: str = "train") -> None:
        """Сохранение metadata.json.
        
        Args:
            samples: Список образцов
            output_path: Путь к metadata.json
            split_name: Имя split (train или val)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            metadata = []
            for idx, sample in enumerate(samples):
                metadata.append({
                    'image_id': f"{idx+1:05d}",
                    'image_path': f"images/{idx+1:05d}.jpg",
                    'text': sample['text'],
                    'original_filename': sample['filename'],
                    'width': sample.get('processed_width', sample.get('width')),
                    'height': sample.get('processed_height', sample.get('height')),
                    'split': split_name
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Metadata сохранена: {output_path} ({len(metadata)} записей)")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения metadata: {e}", exc_info=True)
            raise
