#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning DeepSeek-OCR с LoRA

Автор: Николай
Дата: 2026-01-08
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
from utils.trainer import LoRATrainer


def main(args):
    """Главная функция.
    
    Args:
        args: Аргументы командной строки
    
    Returns:
        Exit code (0 = успех, 1 = ошибка)
    """
    try:
        # Настройка логгера
        log_file = Path("logs") / f"train_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger("train_lora", log_file)
        
        logger.info("=" * 80)
        logger.info("Запуск обучения LoRA")
        logger.info("=" * 80)
        
        # Создание тренера
        trainer = LoRATrainer(config_path=Path(args.config), logger=logger)
        
        # Подготовка
        logger.info("Шаг 1: Загрузка модели и процессора...")
        trainer.load_model_and_processor()
        
        logger.info("Шаг 2: Настройка LoRA...")
        trainer.setup_lora()
        
        logger.info("Шаг 3: Подготовка датасетов...")
        trainer.prepare_datasets()
        
        logger.info("Шаг 4: Создание Trainer...")
        trainer.create_trainer()
        
        # Обучение
        logger.info("Шаг 5: Запуск обучения...")
        trainer.train()
        
        logger.info("=" * 80)
        logger.info("Обучение завершено успешно!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Ошибка: {e}", exc_info=True)
        else:
            print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning DeepSeek-OCR с LoRA")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Путь к конфигурационному файлу")
    
    args = parser.parse_args()
    sys.exit(main(args))
