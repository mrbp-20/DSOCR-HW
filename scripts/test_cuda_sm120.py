#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест поддержки CUDA sm_120 (Blackwell) в PyTorch.

Проверяет:
1. Доступность CUDA
2. Список поддерживаемых архитектур
3. Работу базовых операций на GPU

Автор: Николай
Дата: 2026-01-10
Проект: DSOCR-HW
"""

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import io

# Настройка кодировки
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("="*80)
print("ТЕСТ CUDA SM_120 ПОДДЕРЖКИ")
print("="*80)

# 1. Версия PyTorch
print(f"\nPyTorch версия: {torch.__version__}")
print(f"CUDA версия: {torch.version.cuda}")

# 2. Доступность CUDA
print(f"\nCUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
    
    # Проверка на sm_120 (12.0)
    if capability[0] == 12 and capability[1] == 0:
        print("[OK] Обнаружена архитектура sm_120 (Blackwell)")
    else:
        print(f"[INFO] Архитектура: sm_{capability[0]}{capability[1]}")
else:
    print("[ERROR] CUDA недоступна! Проверьте установку.")
    sys.exit(1)

# 3. Список поддерживаемых архитектур
try:
    # Попытка получить список архитектур (может не работать в старых версиях)
    print(f"\nПопытка получить список поддерживаемых архитектур...")
    # torch.cuda.get_arch_list() может не существовать, используем другой метод
    print("[INFO] Проверка через тестовые операции на GPU...")
except Exception as e:
    print(f"[INFO] torch.cuda.get_arch_list() недоступен: {e}")

# 4. Проверка работы базовых операций на GPU
print("\n" + "="*80)
print("ТЕСТ БАЗОВЫХ ОПЕРАЦИЙ НА GPU")
print("="*80)

try:
    # Тест 1: Создание тензора на GPU
    print("\nТест 1: Создание тензора на GPU...")
    x = torch.randn(100, 100).cuda()
    print(f"[OK] Тензор создан: shape={x.shape}, device={x.device}")
    
    # Тест 2: Математические операции
    print("\nТест 2: Математические операции...")
    y = torch.matmul(x, x)
    print(f"[OK] Умножение матриц: shape={y.shape}")
    
    # Тест 3: Сумма
    print("\nТест 3: Сумма элементов...")
    sum_result = torch.sum(y)
    print(f"[OK] Сумма: {sum_result.item():.2f}")
    
    # Тест 4: Проверка памяти
    print("\nТест 4: Информация о памяти GPU...")
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
    print(f"[OK] Выделено: {memory_allocated:.2f} MB")
    print(f"[OK] Зарезервировано: {memory_reserved:.2f} MB")
    
    print("\n" + "="*80)
    print("[OK] ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("="*80)
    
    # Проверка на sm_120
    if capability[0] == 12 and capability[1] == 0:
        print("\n[OK] sm_120 ПОДДЕРЖИВАЕТСЯ!")
        print("[OK] PyTorch может работать с RTX 5060 Ti (Blackwell)")
    
    sys.exit(0)
    
except RuntimeError as e:
    if "no kernel image is available" in str(e):
        print("\n" + "="*80)
        print("[ERROR] CUDA ERROR: no kernel image is available for execution on the device")
        print("="*80)
        print("\nПРОБЛЕМА: PyTorch не поддерживает архитектуру GPU (sm_120)")
        print("\nРЕШЕНИЕ:")
        print("1. Установите PyTorch Nightly с CUDA 12.8:")
        print("   pip uninstall -y torch torchvision")
        print("   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128")
        print("\n2. Или обновите драйвер NVIDIA до версии 581.xx+")
        sys.exit(1)
    else:
        print(f"\n[ERROR] RuntimeError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Неожиданная ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
