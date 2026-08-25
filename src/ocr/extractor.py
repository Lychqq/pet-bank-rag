"""
OCR модуль — извлечение структурированных данных из документов через Gemini vision.

Gemini 2.5 Flash мультимодальный — читает изображения и PDF нативно,
без отдельного OCR-движка (Tesseract). Один вызов = и чтение и понимание структуры.

После извлечения — валидация через обычный Python код (re, datetime),
а не через LLM. Детерминированно, быстро, бесплатно.
"""

import re
import base64
import json
from datetime import datetime
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.config import GOOGLE_API_KEY, GEMINI_MODEL


def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,  # максимально детерминированно для извлечения данных
    )


# =============================================
# Извлечение данных из документа через Gemini vision
# =============================================

def extract_document_data(image_path: str) -> dict:
    """
    Принимает путь к изображению (скан паспорта, договора, и т.д.).
    Gemini сразу читает пиксели и извлекает структурированные данные.

    Почему Gemini, а не Tesseract:
    - Tesseract: пиксели → сырой текст (нужен второй шаг для структуры)
    - Gemini vision: пиксели → структурированные поля сразу (один шаг)
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    # Читаем изображение и кодируем в base64
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Определяем тип файла по расширению
    ext = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    prompt = """Извлеки данные из этого документа и верни их в формате JSON.

Если это паспорт, верни:
{
  "document_type": "passport",
  "full_name": "...",
  "birth_date": "YYYY-MM-DD",
  "series_number": "XXXX XXXXXX",
  "issue_date": "YYYY-MM-DD",
  "issued_by": "..."
}

Если это банковский договор, верни:
{
  "document_type": "contract",
  "contract_number": "...",
  "client_name": "...",
  "product_type": "...",
  "amount": null или число,
  "rate": null или число,
  "term_months": null или число,
  "sign_date": "YYYY-MM-DD"
}

Верни ТОЛЬКО JSON без пояснений и без markdown-блоков."""

    # Отправляем изображение + промпт в Gemini
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                },
            },
            {"type": "text", "text": prompt},
        ]
    )

    response = llm.invoke([message])

    # Парсим JSON из ответа
    raw = response.content.strip()
    # Убираем возможные markdown-блоки если модель их добавила
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Не удалось распарсить ответ модели", "raw": raw}

    return extracted


from src.ocr.validators import validate_passport_data, validate_contract_data



# =============================================
# Главная функция — извлечение + валидация
# =============================================

def process_document(image_path: str) -> dict:
    """
    Полный пайплайн обработки документа:
    1. Gemini vision извлекает структурированные данные (LLM — "мягкая" работа)
    2. Python-валидация проверяет форматы и бизнес-правила (код — "жёсткая" работа)

    Возвращает словарь с данными, ошибками и статусом.
    """
    extracted = extract_document_data(image_path)

    if "error" in extracted:
        return {
            "status": "extraction_failed",
            "error": extracted["error"],
            "data": None,
            "validation_errors": [],
        }

    doc_type = extracted.get("document_type", "unknown")
    validation_errors = []

    if doc_type == "passport":
        validation_errors = validate_passport_data(extracted)
    elif doc_type == "contract":
        validation_errors = validate_contract_data(extracted)

    status = "valid" if not validation_errors else "needs_review"

    return {
        "status": status,
        "document_type": doc_type,
        "data": extracted,
        "validation_errors": validation_errors,
        "requires_human_review": len(validation_errors) > 0,
    }
