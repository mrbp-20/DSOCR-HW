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
import torchvision.transforms as transforms
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


class DSModelTrainer(Trainer):
    """
    Кастомный Trainer для DeepSeek-OCR (CausalLM).
    
    Переопределяет _prepare_inputs и compute_loss, чтобы корректно передавать данные в модель.
    DeepSeek-OCR — это CausalLM (как GPT), а не Encoder-Decoder (как T5),
    поэтому не поддерживает decoder_input_ids.
    """
    
    def _prepare_inputs(self, inputs):
        """
        Удаляет decoder_input_ids, который Trainer автоматически добавляет.
        
        Проблема:
            HuggingFace Trainer автоматически добавляет decoder_input_ids для Vision2Seq моделей
            в методе _prepare_inputs(). Это стандартное поведение для EncoderDecoder моделей (T5, BART),
            но DeepSeek-OCR — это CausalLM, который не понимает decoder_input_ids.
        
        Решение:
            Переопределяем _prepare_inputs, чтобы удалить decoder_input_ids из inputs
            ДО того, как они попадут в compute_loss().
        
        Args:
            inputs: Словарь с входными данными (может содержать decoder_input_ids)
        
        Returns:
            inputs без decoder_input_ids
        """
        inputs = super()._prepare_inputs(inputs)
        
        # КРИТИЧНО: DeepSeek-OCR — это CausalLM, не EncoderDecoder
        if "decoder_input_ids" in inputs:
            inputs.pop("decoder_input_ids")  # Удаляем!
        
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom compute_loss для DeepSeek-OCR (CausalLM).
        
        Проблема:
            HuggingFace Trainer автоматически передаёт decoder_input_ids для Vision2Seq,
            но DeepSeek-OCR — это CausalLM (как GPT), а не Encoder-Decoder (как T5).
            CausalLM не поддерживает параметр decoder_input_ids.
        
        Решение:
            Переопределяем compute_loss, чтобы передать в модель правильные параметры:
            - pixel_values (от vision encoder)
            - input_ids (текстовые токены для decoder)
            - labels (для loss computation)
        
        Args:
            model: Модель DeepSeek-OCR с LoRA
            inputs: Batch из data_collator с ключами:
                    - pixel_values: [batch_size, 3, H, W]
                    - input_ids: [batch_size, seq_len]
                    - attention_mask: [batch_size, seq_len]
                    - labels: [batch_size, seq_len]
            return_outputs: Вернуть outputs вместе с loss
        
        Returns:
            loss (и outputs, если return_outputs=True)
        """
        # 1. Удаляем decoder_input_ids на всякий случай (если он всё же попал сюда)
        if "decoder_input_ids" in inputs:
            inputs.pop("decoder_input_ids")
        
        # 2. Создаём новый словарь ТОЛЬКО с нужными параметрами
        # Это гарантирует, что decoder_input_ids не попадёт в модель
        model_inputs = {}
        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
        if "input_ids" in inputs:
            model_inputs["input_ids"] = inputs["input_ids"]
        if "attention_mask" in inputs:
            model_inputs["attention_mask"] = inputs["attention_mask"]
        if "labels" in inputs:
            model_inputs["labels"] = inputs["labels"]
        
        # 3. Вызываем forward pass модели с правильными параметрами
        # DeepSeek-OCR принимает:
        # - pixel_values: изображения для vision encoder
        # - input_ids: текстовые токены (как в GPT)
        # - attention_mask: маска для input_ids
        # - labels: целевые токены для loss
        outputs = model(**model_inputs)
        
        # 3. Извлекаем loss из outputs
        # Модель автоматически считает CrossEntropyLoss между predictions и labels
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


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
            
            # Импорт wrapper для фильтрации decoder_input_ids
            from utils.model_wrapper import DeepSeekOCRWrapper
            
            # Создание конфигурации LoRA
            # ВАЖНО: используем SEQ_2_SEQ_LM (не CAUSAL_LM!), т.к. DeepSeek-OCR имеет vision encoder + text decoder
            lora_config = LoraConfig(
                r=lora_config_dict['r'],
                lora_alpha=lora_config_dict['lora_alpha'],
                target_modules=lora_config_dict['target_modules'],
                lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
                bias=lora_config_dict.get('bias', 'none'),
                task_type=TaskType.SEQ_2_SEQ_LM  # Vision2Seq модель (vision encoder + text decoder)
            )
            
            # КРИТИЧНО: Оборачиваем модель ПЕРЕД применением LoRA
            # Это фильтрует decoder_input_ids, который PEFT автоматически добавляет
            self.logger.info("Оборачиваем модель для совместимости с PEFT...")
            self.model = DeepSeekOCRWrapper(self.model)
            
            # Применение LoRA к wrapper (не к базовой модели)
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
        """
        Data collator для DeepSeek-OCR с раздельной обработкой images и text.
        
        DeepSeek-OCR требует:
        1. Ручная предобработка изображений через torchvision.transforms
           (стандартный ImageNet preprocessing: resize + normalize)
        2. Tokenizer для текста (через processor.batch_encode_plus)
        
        Args:
            examples: Список словарей с ключами 'image_path' и 'text'
        
        Returns:
            Батч для обучения с ключами:
            - pixel_values: тензор изображений [batch_size, 3, H, W]
            - input_ids: токенизированный текст
            - attention_mask: маска внимания
            - labels: метки для loss (копия input_ids)
        """
        # 1. Загрузка изображений
        images = [Image.open(ex['image_path']).convert('RGB') for ex in examples]
        texts = [ex['text'] for ex in examples]
        
        # 2. РУЧНАЯ ПРЕДОБРАБОТКА IMAGES через torchvision
        # DeepSeek-OCR использует стандартный ImageNet preprocessing
        try:
            # Получаем размер изображения из конфига (или используем дефолт 1024)
            data_config = self.config.get('data', {})
            preprocessing_config = data_config.get('preprocessing', {})
            image_size = preprocessing_config.get('image_size', 1024)
            
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # Resize до квадрата
                transforms.ToTensor(),  # Конвертируем в тензор [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean (стандарт для Vision Transformers)
                    std=[0.229, 0.224, 0.225]    # ImageNet std
                )
            ])
            
            # Применяем transforms к каждому изображению и собираем в batch
            pixel_values = torch.stack([transform(img) for img in images])
            
            self.logger.debug(f"Обработано {len(images)} изображений, pixel_values shape: {pixel_values.shape}")
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки изображений: {e}", exc_info=True)
            raise
        
        # 3. Токенизация текста через processor (уже работает!)
        try:
            data_config = self.config.get('data', {})
            max_length = data_config.get('max_seq_length', 512)
            
            text_inputs = self.processor.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            self.logger.debug(f"Токенизировано {len(texts)} текстов, input_ids shape: {text_inputs['input_ids'].shape}")
            
        except Exception as e:
            self.logger.error(f"Ошибка токенизации текста: {e}", exc_info=True)
            raise
        
        # 4. Объединяем все inputs в один батч
        batch = {
            'pixel_values': pixel_values,  # [batch_size, 3, H, W]
            'input_ids': text_inputs['input_ids'],  # [batch_size, seq_len]
            'attention_mask': text_inputs['attention_mask'],  # [batch_size, seq_len]
        }
        
        # 5. Labels = input_ids для teacher forcing (стандарт для seq2seq)
        # Копируем, чтобы не изменять оригинальный тензор
        batch['labels'] = text_inputs['input_ids'].clone()
        
        return batch
    
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
            
            # Создание Trainer (используем кастомный DSModelTrainer с compute_loss)
            self.trainer = DSModelTrainer(
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
