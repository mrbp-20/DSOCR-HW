#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper для DeepSeek-OCR модели для совместимости с PEFT.

Проблема:
    PEFT автоматически добавляет decoder_input_ids для Vision2Seq моделей,
    но DeepSeek-OCR (CausalLM) его не понимает.

Решение:
    Wrapper принимает decoder_input_ids (чтобы PEFT был доволен),
    но не передаёт его в базовую модель.

Автор: Николай
Дата: 2026-01-10
Проект: DSOCR-HW
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class DeepSeekOCRWrapper(nn.Module):
    """
    Wrapper для DeepSeek-OCR, который фильтрует decoder_input_ids.
    
    PEFT автоматически добавляет decoder_input_ids для Vision2Seq моделей,
    но DeepSeek-OCR (CausalLM) его не понимает. Этот wrapper решает проблему.
    
    Usage:
        # Загружаем базовую модель
        base_model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", ...)
        
        # Оборачиваем ПЕРЕД применением LoRA
        model = DeepSeekOCRWrapper(base_model)
        
        # Применяем PEFT/LoRA к wrapper
        model = get_peft_model(model, lora_config)
    """
    
    def __init__(self, base_model):
        """
        Args:
            base_model: Базовая модель DeepSeek-OCR (до применения PEFT)
        """
        super().__init__()
        self.model = base_model
        
        # Копируем важные атрибуты, чтобы PEFT корректно работал
        self.config = base_model.config
        self.base_model = base_model  # PEFT ищет этот атрибут
        
        # Копируем методы генерации для inference
        if hasattr(base_model, 'generate'):
            self.generate = base_model.generate
        if hasattr(base_model, 'prepare_inputs_for_generation'):
            self.prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,  # ПРИНИМАЕМ, но ИГНОРИРУЕМ
        **kwargs  # Ловим все остальные параметры
    ) -> Union[Tuple, torch.Tensor]:
        """
        Forward pass с фильтрацией decoder_input_ids.
        
        PEFT может передать decoder_input_ids — мы его просто игнорируем
        и передаём в модель только то, что она понимает.
        
        Args:
            pixel_values: Изображения (от vision encoder)
            input_ids: Текстовые токены (для CausalLM)
            attention_mask: Маска для input_ids
            labels: Целевые токены для loss
            decoder_input_ids: ИГНОРИРУЕТСЯ (для совместимости с PEFT)
            **kwargs: Дополнительные параметры (тоже игнорируются)
        
        Returns:
            Outputs от базовой модели (с loss, logits, etc.)
        """
        # Передаём в базовую модель ТОЛЬКО то, что она понимает
        # decoder_input_ids НЕ передаём!
        # ВАЖНО: DeepSeek-OCR использует 'images', не 'pixel_values'!
        return self.model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def __getattr__(self, name):
        """
        Прокси для всех остальных атрибутов базовой модели.
        
        Если PEFT или Trainer запрашивают атрибут, которого нет в wrapper,
        пробрасываем запрос к базовой модели.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
