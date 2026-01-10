#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Trainer for DSOCR-HW project.

Автор: Николай
Дата: 2026-01-08
Проект: DSOCR-HW
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from PIL import Image
import numpy as np

from transformers import (
    AutoModel,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml
from tqdm import tqdm


class LoRATrainer:
    """Класс для fine-tuning DeepSeek-OCR с LoRA.
    
    Attributes:
        config: Конфигурация обучения из YAML
        logger: Логгер для логирования
        model: Загруженная модель
        processor: Процессор (tokenizer + image processor)
        train_dataset: Обучающий датасет
        eval_dataset: Валидационный датасет
    """
    
    def __init__(self, config_path: Path, logger: Optional[logging.Logger] = None):
        """Инициализация тренера.
        
        Args:
            config_path: Путь к конфигурационному файлу (YAML)
            logger: Логгер (опционально)
        """
        self.config_path = Path(config_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Загрузка конфигурации
        self._load_config()
        
        # Инициализация компонентов (будут загружены при вызове prepare)
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
    
    def _load_config(self) -> None:
        """Загрузка конфигурации из YAML файла."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Конфигурация загружена из {self.config_path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}", exc_info=True)
            raise
    
    def load_model_and_processor(self) -> None:
        """Загрузка модели и процессора с индикатором прогресса."""
        try:
            self.logger.info("Загрузка модели и процессора...")
            
            model_config = self.config['model']
            base_model = model_config['base_model']
            
            # Revision для DeepSeek-OCR (совместимая версия)
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
                self.logger.info(f"Процессор загружен: {base_model} (revision: {revision})")
                
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
    
    def setup_lora(self) -> None:
        """Настройка LoRA адаптера."""
        try:
            self.logger.info("Настройка LoRA...")
            
            if self.model is None:
                raise ValueError("Модель не загружена. Вызовите load_model_and_processor() сначала.")
            
            lora_config_dict = self.config['lora']
            
            # Создание конфигурации LoRA
            lora_config = LoraConfig(
                r=lora_config_dict['r'],
                lora_alpha=lora_config_dict['lora_alpha'],
                target_modules=lora_config_dict['target_modules'],
                lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
                bias=lora_config_dict.get('bias', 'none'),
                task_type=TaskType[lora_config_dict.get('task_type', 'SEQ_2_SEQ_LM')]
            )
            
            # Применение LoRA к модели
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.logger.info("LoRA настроен успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки LoRA: {e}", exc_info=True)
            raise
    
    def _load_metadata(self, metadata_path: Path) -> List[Dict[str, Any]]:
        """Загрузка metadata.json.
        
        Args:
            metadata_path: Путь к metadata.json
        
        Returns:
            Список записей из metadata
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            self.logger.error(f"Ошибка загрузки metadata: {e}", exc_info=True)
            raise
    
    def _process_dataset_item(self, item: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """Обработка одного элемента датасета.
        
        Args:
            item: Элемент из metadata
            base_path: Базовый путь к директории с данными
        
        Returns:
            Обработанный элемент с изображением и текстом
        """
        try:
            image_path = base_path / item['image_path']
            image = Image.open(image_path).convert('RGB')
            text = item['text']
            
            return {
                'image': image,
                'text': text
            }
        except Exception as e:
            self.logger.error(f"Ошибка обработки элемента {item.get('image_id', 'unknown')}: {e}", exc_info=True)
            raise
    
    def _data_collator(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Data collator для обработки батчей данных.
        
        Args:
            examples: Список примеров из датасета
        
        Returns:
            Словарь с обработанными данными (pixel_values, input_ids, attention_mask, labels)
        """
        try:
            # Загрузка изображений
            images = [Image.open(ex['image_path']).convert('RGB') for ex in examples]
            texts = [ex['text'] for ex in examples]
            
            # Обработка через processor
            # Processor возвращает pixel_values, input_ids, attention_mask
            # Для обучения нужны labels (input_ids для targets)
            inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
            
            # Labels = input_ids для sequence-to-sequence
            if 'input_ids' in inputs:
                inputs['labels'] = inputs['input_ids'].clone()
            
            return inputs
        except Exception as e:
            self.logger.error(f"Ошибка в data_collator: {e}", exc_info=True)
            raise
    
    def prepare_datasets(self) -> None:
        """Подготовка датасетов для обучения с индикатором прогресса."""
        try:
            self.logger.info("Подготовка датасетов...")
            
            data_config = self.config['data']
            train_path = Path(data_config['train_path'])
            val_path = Path(data_config['val_path'])
            
            # Прогресс-бар для загрузки данных
            with tqdm(total=2, desc="Загрузка данных", unit="split", ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                # Загрузка train metadata
                pbar.set_description("Загрузка train данных")
                train_metadata = self._load_metadata(train_path / 'metadata.json')
                pbar.update(1)
                
                # Загрузка val metadata
                pbar.set_description("Загрузка val данных")
                val_metadata = self._load_metadata(val_path / 'metadata.json')
                pbar.update(1)
            
            self.logger.info(f"Загружено train: {len(train_metadata)}, val: {len(val_metadata)} образцов")
            
            # Сохраняем пути и тексты (изображения загружаются в data_collator)
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
    
    def create_trainer(self) -> None:
        """Создание Trainer объекта."""
        try:
            self.logger.info("Создание Trainer...")
            
            if self.model is None or self.train_dataset is None or self.eval_dataset is None:
                raise ValueError("Модель и датасеты должны быть подготовлены")
            
            training_config = self.config['training']
            optimization_config = self.config.get('optimization', {})
            
            # TrainingArguments
            training_args = TrainingArguments(
                output_dir=training_config['output_dir'],
                num_train_epochs=training_config['num_train_epochs'],
                per_device_train_batch_size=training_config['per_device_train_batch_size'],
                per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
                gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
                learning_rate=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 0.01),
                warmup_steps=training_config.get('warmup_steps', 0),
                logging_steps=training_config.get('logging_steps', 10),
                save_steps=training_config.get('save_steps', 500),
                save_total_limit=training_config.get('save_total_limit', 3),
                eval_steps=training_config.get('eval_steps', 500),
                evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
                save_strategy=training_config.get('save_strategy', 'steps'),
                load_best_model_at_end=training_config.get('load_best_model_at_end', False),
                metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
                greater_is_better=training_config.get('greater_is_better', False),
                fp16=training_config.get('fp16', False),
                dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
                remove_unused_columns=training_config.get('remove_unused_columns', False),
                report_to=training_config.get('report_to', []),
                logging_dir=training_config.get('logging_dir'),
                seed=training_config.get('seed', 42),
                lr_scheduler_type=optimization_config.get('lr_scheduler_type', 'linear'),
                max_grad_norm=optimization_config.get('max_grad_norm', 1.0),
                disable_tqdm=False  # Включить progress bars во время обучения
            )
            
            # Callbacks
            callbacks = []
            if self.config.get('early_stopping'):
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping'].get('patience', 5),
                    early_stopping_threshold=self.config['early_stopping'].get('threshold', 0.01)
                )
                callbacks.append(early_stopping)
            
            # Создание Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=self._data_collator,
                callbacks=callbacks
            )
            
            self.logger.info("Trainer создан")
            
        except Exception as e:
            self.logger.error(f"Ошибка создания Trainer: {e}", exc_info=True)
            raise
    
    def train(self) -> None:
        """Запуск обучения."""
        try:
            if self.trainer is None:
                raise ValueError("Trainer не создан. Вызовите create_trainer() сначала.")
            
            self.logger.info("=" * 80)
            self.logger.info("Начало обучения")
            self.logger.info("=" * 80)
            
            # Обучение
            self.trainer.train()
            
            # Сохранение финальной модели
            self.logger.info("Сохранение модели...")
            self.trainer.save_model()
            
            self.logger.info("=" * 80)
            self.logger.info("Обучение завершено успешно!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}", exc_info=True)
            raise
