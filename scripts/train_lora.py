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
from utils.encoding_fix import fix_windows_console_encoding


def main(args):
    """Главная функция.
    
    Args:
        args: Аргументы командной строки
    
    Returns:
        Exit code (0 = успех, 1 = ошибка)
    """
    # ========================================
    # КРИТИЧНО: Установка UTF-8 для Windows
    # ========================================
    fix_windows_console_encoding()
    
    try:
        # Настройка логгера
        log_file = Path("logs") / f"train_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger("train_lora", log_file)
        
        logger.info("=" * 80)
        logger.info("Запуск обучения LoRA")
        logger.info("=" * 80)
        
        # Создание тренера
        trainer = LoRATrainer(config_path=Path(args.config), logger=logger)
        
        # Шаг 1: Загрузка модели
        logger.info("\n" + "=" * 80)
        logger.info("Шаг 1/5: Загрузка модели и процессора...")
        logger.info("=" * 80)
        trainer.load_model_and_processor()
        logger.info("OK Шаг 1 завершён")
        
        # Шаг 2: LoRA
        logger.info("\n" + "=" * 80)
        logger.info("Шаг 2/5: Настройка LoRA...")
        logger.info("=" * 80)
        trainer.setup_lora()
        logger.info("OK Шаг 2 завершён")
        
        # Шаг 3: Датасеты
        logger.info("\n" + "=" * 80)
        logger.info("Шаг 3/5: Подготовка датасетов...")
        logger.info("=" * 80)
        trainer.prepare_datasets()
        logger.info("OK Шаг 3 завершён")
        
        # Шаг 4: Trainer
        logger.info("\n" + "=" * 80)
        logger.info("Шаг 4/5: Создание Trainer...")
        logger.info("=" * 80)
        trainer.create_trainer()
        logger.info("OK Шаг 4 завершён")
        
        # Шаг 5: Обучение
        logger.info("\n" + "=" * 80)
        logger.info("Шаг 5/5: ЗАПУСК ОБУЧЕНИЯ!")
        logger.info("=" * 80)
        trainer.train()
        logger.info("OK Шаг 5 завершён")
        
        logger.info("\n" + "=" * 80)
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
