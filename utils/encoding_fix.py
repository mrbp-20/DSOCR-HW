#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для исправления кодировки консоли в Windows.

Проблема: Windows по умолчанию использует CP1251/CP866,
что приводит к крякозяблам при выводе русского текста и emoji.

Решение: Принудительная установка UTF-8 для всех потоков вывода.
"""

import sys
import os


def fix_windows_console_encoding():
    """
    Исправляет кодировку консоли для корректного отображения UTF-8 в Windows.
    
    Выполняет:
    1. Переконфигурирует stdout/stderr на UTF-8
    2. Устанавливает PYTHONIOENCODING=utf-8
    3. Устанавливает Windows Console Code Page на 65001 (UTF-8)
    
    Безопасно для других ОС — ничего не делает на Linux/macOS.
    
    Usage:
        from utils.encoding_fix import fix_windows_console_encoding
        
        # В начале main():
        fix_windows_console_encoding()
        print("Теперь можно использовать русский текст и emoji! 🚀")
    """
    # Проверяем, что это Windows
    if sys.platform != 'win32':
        return  # На Linux/macOS ничего не делаем
    
    # 1. Переконфигурируем stdout и stderr
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Для старых версий Python (< 3.7)
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 2. Устанавливаем переменную окружения для подпроцессов
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 3. Устанавливаем Windows Console Code Page на UTF-8 (65001)
    try:
        import ctypes
        # SetConsoleCP - для ввода
        ctypes.windll.kernel32.SetConsoleCP(65001)
        # SetConsoleOutputCP - для вывода
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        # Игнорируем ошибки (может не работать в некоторых терминалах)
        pass


def print_encoding_info():
    """
    Выводит информацию о текущих кодировках для отладки.
    
    Usage:
        from utils.encoding_fix import print_encoding_info
        print_encoding_info()
    """
    print("=" * 60)
    print("📊 Информация о кодировках:")
    print("=" * 60)
    print(f"sys.stdout.encoding: {sys.stdout.encoding}")
    print(f"sys.stderr.encoding: {sys.stderr.encoding}")
    print(f"sys.getdefaultencoding(): {sys.getdefaultencoding()}")
    print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'не установлено')}")
    print(f"Платформа: {sys.platform}")
    
    # Тест вывода
    print("\n🧪 Тест вывода:")
    print("  - Русский текст: Привет, мир!")
    print("  - Emoji: 🚀 💪 ✅ ❌ 🎯")
    print("  - Progress bar simulation: " + "█" * 20)
    print("=" * 60)
