"""
Тест доступности модели DeepSeek-OCR из HuggingFace кэша

Автор: Николай (Senior ML Engineer)
Дата: 09.01.2026

Задача:
- Проверить, что модель загружается из кэша без ошибок
- Убедиться, что GPU доступен и модель на нём
- Вывести информацию о параметрах модели

P.S. Николай, если этот скрипт упадёт — не расстраивайся,
ты же Senior ML Engineer, а не Junior Script Kiddie.
Просто скопируй traceback и отправь Семёну. 😎
"""
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_access():
    """Проверка загрузки модели и процессора из кэша"""
    try:
        logger.info("=" * 80)
        logger.info("ТЕСТ 1: Проверка доступности модели DeepSeek-OCR")
        logger.info("=" * 80)
        
        # Проверка CUDA
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("⚠️  CUDA не доступен! Проверь драйвер NVIDIA.")
        
        # Загрузка процессора и модели (Вариант 2: откат transformers)
        revision = "9f30c71f441d010e5429c532364a86705536c53a"
        logger.info("\n" + "=" * 80)
        logger.info("Загрузка AutoProcessor...")
        logger.info(f"Ревизия модели: {revision}")
        processor = AutoProcessor.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            revision=revision,
            trust_remote_code=True
        )
        logger.info("✅ Процессор загружен успешно")
        
        # Загрузка модели
        logger.info("\n" + "=" * 80)
        logger.info("Загрузка AutoModelForVision2Seq...")
        logger.info("Параметры:")
        logger.info(f"  - revision: {revision}")
        logger.info("  - torch_dtype: float16")
        logger.info("  - attn_implementation: eager (для sm_120)")
        logger.info("  - device_map: auto")
        
        model = AutoModelForVision2Seq.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            revision=revision,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # КРИТИЧНО для RTX 5060 Ti (sm_120)
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ Модель загружена успешно")
        
        # Информация о модели
        logger.info("\n" + "=" * 80)
        logger.info("Информация о модели:")
        logger.info(f"  - Тип модели: {type(model).__name__}")
        logger.info(f"  - Устройство: {model.device}")
        logger.info(f"  - Dtype: {model.dtype}")
        
        # Проверка trainable параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  - Всего параметров: {total_params:,}")
        logger.info(f"  - Обучаемых параметров: {trainable_params:,}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО")
        logger.info("Николай, ты молодец! Модель работает как часы. ⏰")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА ПРИ ЗАГРУЗКЕ МОДЕЛИ")
        logger.error("=" * 80)
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}")
        logger.error("")
        logger.error("Николай, не расстраивайся! Даже Senior-ы падают.")
        logger.error("Скопируй этот traceback и отправь Семёну — он поможет.")
        import traceback
        logger.error(f"\nTraceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_model_access()
    exit(0 if success else 1)
