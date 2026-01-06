# DSOCR-HW: DeepSeek-OCR Handwriting Fine-tuning

**Проект для fine-tuning DeepSeek-OCR на рукописный текст с использованием LoRA**

## Цель проекта

Создать лёгкий LoRA-адаптер для базовой модели DeepSeek-OCR, который обучен распознавать **личные рукописные документы** (заметки, архивы, формы). Этот адаптер решает задачи, недоступные классическим OCR-системам (Tesseract), которые плохо работают с рукописным текстом.

## Команда

- **Владимир** (Product Owner) — определяет задачи, контролирует качество датасета, финальный ревью
- **Семён "Сёма"** (Tech Lead) — архитектура проекта, координация, code review, документация
- **Николай** (Senior ML Engineer, Cursor AI) — имплементация скриптов, fine-tuning, эксперименты

## Железо и окружение

- **GPU**: NVIDIA GeForce RTX 5060 Ti 16 ГБ (CUDA Capability sm_120)
- **OS**: Windows 10
- **Python**: 3.10+ (в venv)
- **PyTorch**: 2.9.1+cu128 (для RTX 5060 Ti)
- **IDE**: Cursor AI

## Структура проекта

```
DSOCR-HW/
├── data/                          # Датасеты
│   ├── raw/                       # Сырые сканы рукописей
│   ├── annotated/                 # Размеченные данные (annotations.csv)
│   └── processed/                 # Обработанные для тренировки
│       ├── train/
│       └── val/
├── models/                        # Модели и адаптеры
│   ├── base/                      # Базовая DeepSeek-OCR (кэш HuggingFace)
│   └── lora_adapters/             # Натренированные LoRA
│       └── handwriting_v1/
├── scripts/                       # Скрипты обработки
│   ├── prepare_dataset.py         # Подготовка датасета
│   ├── train_lora.py              # Тренировка LoRA
│   ├── evaluate.py                # Оценка CER/WER
│   └── inference.py               # Инференс с LoRA
├── utils/                         # Утилиты
│   ├── image_processor.py         # Предобработка изображений
│   └── logger.py                  # Логирование
├── configs/                       # Конфигурация
│   ├── training_config.yaml       # Параметры тренировки
│   └── inference_config.yaml      # Параметры инференса
├── logs/                          # Логи тренировок и ошибок
├── notebooks/                     # Jupyter для экспериментов
├── .cursorrules                   # Правила для Cursor AI (Николай)
├── requirements.txt               # Python зависимости
├── .gitignore                     # Игнорируемые файлы
└── README.md                      # Этот файл
```

## Быстрый старт для Николая (Cursor AI)

### 1. Клонирование репозитория

```powershell
cd C:\
git clone https://github.com/mrbp-20/DSOCR-HW.git
cd DSOCR-HW
```

### 2. Создание виртуального окружения

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Установка зависимостей

```powershell
pip install -r requirements.txt
```

**Важно**: Если возникают проблемы с Flash Attention (для sm_120), используется `attn_implementation="eager"` в конфиге модели.

### 4. Проверка окружения

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
```

Ожидаемый вывод:
```
PyTorch: 2.9.1+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5060 Ti
```

### 5. Открыть в Cursor

```powershell
cursor .
```

Когда Cursor откроется, он автоматически прочитает `.cursorrules` и будет знать контекст проекта.

## Roadmap (Issues)

- **Issue #1**: Подготовка датасета (`prepare_dataset.py`) — загрузка, валидация, split
- **Issue #2**: Fine-tuning скрипт (`train_lora.py`) — настройка LoRA, training loop
- **Issue #3**: Evaluation скрипт (`evaluate.py`) — метрики CER/WER
- **Issue #4**: Inference скрипт (`inference.py`) — распознавание новых рукописей
- **Issue #5**: Документация и примеры использования

## Workflow

1. **Владимир** сканирует рукописи → `data/raw/`
2. **Владимир** размечает → `data/annotated/annotations.csv`
3. **Николай** пишет скрипты → Pull Request → **Семён** ревьюит → мерж
4. **Николай** запускает тренировку → ночь на GPU → утром результаты
5. **Владимир** тестирует → feedback → итерации

## Важные команды

### Подготовка датасета
```powershell
python scripts/prepare_dataset.py --input data/raw --annotations data/annotated/annotations.csv --output data/processed
```

### Тренировка LoRA
```powershell
python scripts/train_lora.py --config configs/training_config.yaml
```

### Evaluation
```powershell
python scripts/evaluate.py --model models/lora_adapters/handwriting_v1 --test data/processed/val
```

### Inference
```powershell
python scripts/inference.py --model models/lora_adapters/handwriting_v1 --input path/to/handwriting.jpg
```

## Логирование

Все операции логируются в `logs/`:
- `logs/training_YYYYMMDD_HHMMSS.log` — логи тренировок
- `logs/errors_YYYYMMDD_HHMMSS.log` — ошибки
- `logs/inference_YYYYMMDD_HHMMSS.log` — результаты инференса

## Метрики качества

- **CER (Character Error Rate)**: % ошибочных символов
  - CER < 10% — хорошо
  - CER < 5% — отлично
- **WER (Word Error Rate)**: % ошибочных слов

## Troubleshooting

### Flash Attention не устанавливается

В `configs/training_config.yaml` и `configs/inference_config.yaml` установить:
```yaml
model:
  attn_implementation: "eager"  # вместо "flash_attention_2"
```

### CUDA out of memory

Уменьшить `batch_size` в `configs/training_config.yaml`:
```yaml
training:
  batch_size: 2  # вместо 4
```

### PyTorch не видит GPU

Проверить NVIDIA Driver и CUDA Toolkit:
```powershell
nvidia-smi
nvcc --version
```

## Ссылки

- [DeepSeek-OCR на HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [PEFT (LoRA) документация](https://huggingface.co/docs/peft)
- [Unsloth для ускорения fine-tuning](https://github.com/unslothai/unsloth)

## Лицензия

MIT License

---

**Создано командой Dream Team: Владимир, Семён, Николай** 🚀
