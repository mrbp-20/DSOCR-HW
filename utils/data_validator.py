#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data validator for DSOCR-HW project.

Автор: Николай
Дата: 2026-01-08
Проект: DSOCR-HW
"""

from typing import List, Dict, Tuple, Any
import logging
from typing import Optional


class DataValidator:
    """Валидатор датасета.
    
    Проверяет:
    - Наличие обязательных полей
    - Корректность форматов
    - Минимальное качество изображений
    - Корректность текстовых аннотаций
    
    Attributes:
        min_image_size: Минимальный размер изображения (пиксели)
        max_text_length: Максимальная длина текста (символы)
        logger: Логгер для логирования
    """
    
    def __init__(self, min_image_size: int = 100, max_text_length: int = 10000, logger: Optional[logging.Logger] = None):
        """Инициализация валидатора.
        
        Args:
            min_image_size: Минимальный размер изображения (пиксели)
            max_text_length: Максимальная длина текста (символы)
            logger: Логгер (опционально)
        """
        self.min_image_size = min_image_size
        self.max_text_length = max_text_length
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Валидация одного образца.
        
        Args:
            sample: Словарь с ключами 'filename', 'text', 'image', 'width', 'height'
        
        Returns:
            Tuple (is_valid, error_message)
        """
        try:
            # Проверка обязательных полей
            required_fields = ['filename', 'text', 'image', 'width', 'height']
            for field in required_fields:
                if field not in sample:
                    return False, f"Missing field: {field}"
            
            # Проверка размера изображения
            if sample['width'] < self.min_image_size or sample['height'] < self.min_image_size:
                return False, f"Image too small: {sample['width']}x{sample['height']}"
            
            # Проверка текста
            text = str(sample['text']).strip()
            if len(text) == 0:
                return False, "Empty text"
            if len(text) > self.max_text_length:
                return False, f"Text too long: {len(text)} chars"
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации образца: {e}", exc_info=True)
            return False, f"Validation error: {e}"
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Валидация всего датасета.
        
        Args:
            dataset: Список образцов
        
        Returns:
            Отфильтрованный список валидных образцов
        
        Raises:
            ValueError: Если датасет пустой после валидации
        """
        try:
            valid_samples = []
            invalid_count = 0
            
            for sample in dataset:
                is_valid, error = self.validate_sample(sample)
                if is_valid:
                    valid_samples.append(sample)
                else:
                    self.logger.warning(f"Невалидный образец {sample.get('filename', 'unknown')}: {error}")
                    invalid_count += 1
            
            if len(valid_samples) == 0:
                raise ValueError("No valid samples found in dataset")
            
            self.logger.info(f"Валидация завершена: {len(valid_samples)} валидных, {invalid_count} невалидных")
            return valid_samples
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации датасета: {e}", exc_info=True)
            raise
