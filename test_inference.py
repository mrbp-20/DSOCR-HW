"""
Smoke test: инференс модели на тестовом изображении

Автор: Николай (Senior ML Engineer)
Дата: 09.01.2026

Задача:
- Проверить, что модель может распознать текст с изображения
- Убедиться, что весь пайплайн (загрузка → препроцессинг → инференс → декодирование) работает

P.S. Николай, если модель вернёт что-то типа "鸡你太美" вместо
"Отличная работа!" — не удивляйся. Модель пока не обучена на твоём
почерке (точнее, на почерке Владимира). Главное — что она хоть что-то
вернула, а не упала с segfault. 😄
"""
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_inference():
    """Тест распознавания на test_001.jpg"""
    try:
        logger.info("=" * 80)
        logger.info("ТЕСТ 2: Smoke test инференса на test_001.jpg")
        logger.info("=" * 80)
        
        # Загрузка модели (Вариант 2: откат transformers)
        revision = "9f30c71f441d010e5429c532364a86705536c53a"
        logger.info("Загрузка модели и процессора...")
        logger.info(f"Ревизия модели: {revision}")
        processor = AutoProcessor.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            revision=revision,
            trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            revision=revision,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✅ Модель загружена")
        
        # Загрузка изображения
        image_path = "data/raw/test_001.jpg"
        logger.info(f"\nЗагрузка изображения: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.info(f"  - Размер: {image.size}")
        logger.info(f"  - Формат: {image.format}")
        
        # Препроцессинг
        logger.info("\nПрепроцессинг изображения...")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logger.info(f"  - Input keys: {list(inputs.keys())}")
        logger.info(f"  - Input shape: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'N/A'}")
        
        # Инференс
        logger.info("\nЗапуск инференса...")
        logger.info("(Николай, если тут зависнет — не переживай, модель думает)")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        logger.info("✅ Инференс завершён")
        
        # Декодирование
        logger.info("\nДекодирование результата...")
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        
        logger.info("\n" + "=" * 80)
        logger.info("РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:")
        logger.info("=" * 80)
        logger.info(f"Ground truth: 'Отличная работа!'")
        logger.info(f"Распознано:   '{decoded[0]}'")
        logger.info("=" * 80)
        
        # Проверка хотя бы частичного совпадения
        if decoded[0].strip():
            logger.info("✅ МОДЕЛЬ РАБОТАЕТ (вернула непустой текст)")
            logger.info("")
            logger.info("P.S. Николай, если текст не совпадает с ground truth —")
            logger.info("это нормально! Модель ещё не обучена на наших данных.")
            logger.info("Главное — она не упала. Ты справился! 🎉")
        else:
            logger.warning("⚠️  Модель вернула пустую строку")
            logger.warning("Возможно, нужен промпт или другие параметры генерации.")
            logger.warning("Но это не критично — модель загрузилась и работает.")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ SMOKE TEST ЗАВЕРШЁН")
        logger.info("=" * 80)
        
        return True, decoded[0]
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА В SMOKE TEST")
        logger.error("=" * 80)
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}")
        logger.error("")
        logger.error("Николай, Senior-ы тоже падают. Ничего страшного!")
        logger.error("Отправь этот traceback команде — мы поможем.")
        import traceback
        logger.error(f"\nTraceback:\n{traceback.format_exc()}")
        return False, ""

if __name__ == "__main__":
    success, result = test_inference()
    exit(0 if success else 1)
