#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger utility for DSOCR-HW project

Автор: Семён
Дата: 2026-01-06
Проект: DSOCR-HW
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Настройка логгера для проекта.
    
    Args:
        name: Имя логгера
        log_file: Путь к файлу лога (опционально)
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Кастомный формат логов
    
    Returns:
        Настроенный logger
    
    Example:
        >>> logger = setup_logger("my_script", Path("logs/my_script.log"))
        >>> logger.info("Начало работы")
    """
    # Создаём логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Удаляем существующие handlers (если были)
    logger.handlers.clear()
    
    # Формат логов
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (если указан путь)
    if log_file is not None:
        # Создаём директорию для логов, если не существует
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TimestampedLogger:
    """Logger с автоматическим добавлением timestamp к имени файла."""
    
    def __init__(self, name: str, log_dir: Path = Path("logs"), level: int = logging.INFO):
        """Инициализация timestamped logger.
        
        Args:
            name: Имя логгера (используется как префикс файла)
            log_dir: Директория для логов
            level: Уровень логирования
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Создаём timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        # Настраиваем logger
        self.logger = setup_logger(name, log_file, level)
    
    def get_logger(self) -> logging.Logger:
        """Получить настроенный logger.
        
        Returns:
            Настроенный logger
        """
        return self.logger
    
    def __getattr__(self, item):
        """Проксируем методы logger (info, debug, warning, error, critical)."""
        return getattr(self.logger, item)


if __name__ == "__main__":
    # Пример использования
    logger = setup_logger("test", Path("logs/test.log"))
    logger.info("Это информационное сообщение")
    logger.warning("Это предупреждение")
    logger.error("Это ошибка")
    
    # Timestamped logger
    ts_logger = TimestampedLogger("timestamped_test")
    ts_logger.info("Логируем с timestamp в имени файла")
