#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image preprocessing utilities for DSOCR-HW

Автор: Семён
Дата: 2026-01-06
Проект: DSOCR-HW
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from typing import Tuple, Optional, Union
import logging


class ImageProcessor:
    """Класс для предобработки изображений рукописей."""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, logger: Optional[logging.Logger] = None):
        """Инициализация процессора изображений.
        
        Args:
            target_size: Целевой размер изображений (width, height)
            logger: Logger для логирования
        """
        self.target_size = target_size
        self.logger = logger or logging.getLogger(__name__)
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Загрузка изображения.
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            PIL Image
        """
        try:
            image = Image.open(image_path).convert("RGB")
            self.logger.debug(f"Загружено изображение: {image_path}, размер: {image.size}")
            return image
        except Exception as e:
            self.logger.error(f"Ошибка загрузки изображения {image_path}: {e}")
            raise
    
    def resize(self, image: Image.Image, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Изменение размера изображения.
        
        Args:
            image: Исходное изображение
            size: Целевой размер (width, height), если None — используется self.target_size
        
        Returns:
            Изображение с изменённым размером
        """
        size = size or self.target_size
        if size is None:
            return image
        
        original_size = image.size
        resized = image.resize(size, Image.Resampling.LANCZOS)
        self.logger.debug(f"Размер изменён: {original_size} → {size}")
        return resized
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Увеличение контраста.
        
        Args:
            image: Исходное изображение
            factor: Коэффициент контраста (1.0 = без изменений)
        
        Returns:
            Изображение с увеличенным контрастом
        """
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
        self.logger.debug(f"Контраст увеличен (factor={factor})")
        return enhanced
    
    def enhance_sharpness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Увеличение резкости.
        
        Args:
            image: Исходное изображение
            factor: Коэффициент резкости (1.0 = без изменений)
        
        Returns:
            Изображение с увеличенной резкостью
        """
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(factor)
        self.logger.debug(f"Резкость увеличена (factor={factor})")
        return enhanced
    
    def enhance_brightness(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Увеличение яркости.
        
        Args:
            image: Исходное изображение
            factor: Коэффициент яркости (1.0 = без изменений)
        
        Returns:
            Изображение с увеличенной яркостью
        """
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(factor)
        self.logger.debug(f"Яркость увеличена (factor={factor})")
        return enhanced
    
    def denoise(self, image: Image.Image, strength: int = 10) -> Image.Image:
        """Шумоподавление.
        
        Args:
            image: Исходное изображение
            strength: Сила шумоподавления
        
        Returns:
            Изображение с подавленным шумом
        """
        # Конвертируем в numpy array
        img_array = np.array(image)
        
        # Применяем Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
        
        # Конвертируем обратно в PIL Image
        denoised_image = Image.fromarray(denoised)
        self.logger.debug(f"Применено шумоподавление (strength={strength})")
        return denoised_image
    
    def to_grayscale(self, image: Image.Image) -> Image.Image:
        """Конвертация в оттенки серого.
        
        Args:
            image: Исходное изображение
        
        Returns:
            Изображение в оттенках серого
        """
        grayscale = image.convert("L")
        self.logger.debug("Конвертировано в оттенки серого")
        return grayscale
    
    def normalize(self, image: Image.Image) -> np.ndarray:
        """Нормализация изображения.
        
        Args:
            image: Исходное изображение
        
        Returns:
            Нормализованный numpy array [0, 1]
        """
        img_array = np.array(image).astype(np.float32) / 255.0
        self.logger.debug("Изображение нормализовано [0, 1]")
        return img_array
    
    def preprocess(
        self,
        image: Union[str, Path, Image.Image],
        resize: bool = True,
        enhance_contrast: bool = True,
        enhance_sharpness: bool = True,
        enhance_brightness: bool = False,
        denoise: bool = False,
        grayscale: bool = False,
        normalize: bool = False
    ) -> Union[Image.Image, np.ndarray]:
        """Полный пайплайн предобработки изображения.
        
        Args:
            image: Путь к изображению или PIL Image
            resize: Изменять размер
            enhance_contrast: Увеличивать контраст
            enhance_sharpness: Увеличивать резкость
            enhance_brightness: Увеличивать яркость
            denoise: Применять шумоподавление
            grayscale: Конвертировать в оттенки серого
            normalize: Нормализовать [0, 1]
        
        Returns:
            Обработанное изображение (PIL Image или numpy array если normalize=True)
        """
        # Загружаем изображение, если передан путь
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Применяем обработку
        if resize and self.target_size is not None:
            image = self.resize(image)
        
        if enhance_contrast:
            image = self.enhance_contrast(image)
        
        if enhance_sharpness:
            image = self.enhance_sharpness(image)
        
        if enhance_brightness:
            image = self.enhance_brightness(image)
        
        if denoise:
            image = self.denoise(image)
        
        if grayscale:
            image = self.to_grayscale(image)
        
        if normalize:
            return self.normalize(image)
        
        return image


if __name__ == "__main__":
    # Пример использования
    from utils.logger import setup_logger
    
    logger = setup_logger("image_processor_test")
    processor = ImageProcessor(target_size=(224, 224), logger=logger)
    
    # Загружаем и обрабатываем изображение
    # processed = processor.preprocess("path/to/image.jpg")
    print("ImageProcessor готов к использованию")
