#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для исследования API DeepSeek-OCR.
Цель: понять, какие параметры принимает model.forward().

Автор: Николай
Дата: 2026-01-10
Проект: DSOCR-HW
"""

import sys
import inspect
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Настройка вывода и кодировки
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 80)
print("ИССЛЕДОВАНИЕ API DeepSeek-OCR")
print("=" * 80)

# Загрузить модель и процессор
model_name = "deepseek-ai/DeepSeek-OCR"
revision = "9f30c71f441d010e5429c532364a86705536c53a"

print(f"\nЗагрузка модели: {model_name} (revision: {revision})")
print("(Это может занять время...)")

processor = AutoProcessor.from_pretrained(
    model_name,
    revision=revision,
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    model_name,
    revision=revision,
    torch_dtype=torch.float16,
    device_map="cpu",  # Для тестирования на CPU (быстрее для исследований)
    trust_remote_code=True
)

print(f"\n[OK] Модель загружена: {type(model)}")
print(f"[OK] Процессор загружен: {type(processor)}")

# Создать тестовое изображение
image = Image.new("RGB", (224, 224), color="white")
text = "Test OCR prompt"

# ============================================================================
# ИССЛЕДОВАНИЕ 1: Что возвращает processor?
# ============================================================================
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 1: Вызов processor(images=..., text=...)")
print("="*80)

try:
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    )
    
    print(f"\n[OK] processor() успешно вызван")
    print(f"\nКлючи в outputs:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: type={type(value)}, value={value}")
            
except Exception as e:
    print(f"[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()
    inputs = {}

# ============================================================================
# ИССЛЕДОВАНИЕ 2: Сигнатура model.forward()
# ============================================================================
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 2: Сигнатура model.forward()")
print("="*80)

try:
    sig = inspect.signature(model.forward)
    print(f"\nПараметры model.forward():")
    for param_name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        print(f"  - {param_name}: {annotation}{default}")
except Exception as e:
    print(f"[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ИССЛЕДОВАНИЕ 3: Попытка вызвать model.forward() с outputs от processor
# ============================================================================
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 3: Попытка model.forward(**inputs)")
print("="*80)

if inputs:
    try:
        # Переводим все тензоры на CPU (для безопасности)
        inputs_cpu = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("\nПопытка вызвать model.forward()...")
        with torch.no_grad():
            outputs = model.forward(**inputs_cpu)
        
        print(f"[OK] model.forward() успешно вызван!")
        print(f"\nТип outputs: {type(outputs)}")
        
        if hasattr(outputs, '__dict__'):
            print(f"\nАтрибуты outputs:")
            for attr in dir(outputs):
                if not attr.startswith('_'):
                    try:
                        value = getattr(outputs, attr)
                        if not callable(value):
                            print(f"  - {attr}: {type(value)}")
                    except:
                        pass
                        
        if hasattr(outputs, 'loss'):
            print(f"\n  - loss: {outputs.loss}")
        if hasattr(outputs, 'logits'):
            print(f"  - logits: shape={outputs.logits.shape if hasattr(outputs.logits, 'shape') else 'N/A'}")
            
    except Exception as e:
        print(f"❌ Ошибка при вызове model.forward(): {e}")
        print(f"\nТип ошибки: {type(e).__name__}")
        import traceback
        traceback.print_exc()
else:
    print("[WARNING] Пропущено (processor() не вернул данные)")

# ============================================================================
# ИССЛЕДОВАНИЕ 4: processor только для images
# ============================================================================
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 4: processor(images=...) без text")
print("="*80)

try:
    inputs_images = processor(images=image, return_tensors="pt")
    print(f"\n[OK] processor(images=...) успешно вызван")
    print(f"\nКлючи в outputs:")
    for key, value in inputs_images.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: type={type(value)}")
except Exception as e:
    print(f"[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# ИССЛЕДОВАНИЕ 5: prepare_inputs_for_generation (если есть)
# ============================================================================
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 5: prepare_inputs_for_generation")
print("="*80)

if hasattr(model, 'prepare_inputs_for_generation'):
    try:
        sig = inspect.signature(model.prepare_inputs_for_generation)
        print(f"\nПараметры prepare_inputs_for_generation():")
        for param_name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            print(f"  - {param_name}: {annotation}{default}")
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
else:
    print("[WARNING] Метод prepare_inputs_for_generation не найден")

# ============================================================================
# ИССЛЕДОВАНИЕ 6: Атрибуты модели
# ============================================================================
print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ 6: Важные атрибуты модели")
print("="*80)

attrs_to_check = ['config', 'device', 'dtype', 'main_input_name']
for attr in attrs_to_check:
    if hasattr(model, attr):
        value = getattr(model, attr)
        print(f"  - {attr}: {value}")

print("\n" + "="*80)
print("[OK] ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
print("="*80)
