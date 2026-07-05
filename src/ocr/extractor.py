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
    llm = get_llm()

    # Читаем изображение и кодируем в base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Определяем тип файла по расширению
    ext = Path(image_path).suffix.lower()
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


# =============================================
# Валидация извлечённых данных — обычный Python код, НЕ LLM
# =============================================

def validate_passport_data(data: dict) -> list[str]:
    """Валидирует данные паспорта РФ. Возвращает список ошибок (пустой = всё ок)."""
    errors = []

    series_number = data.get("series_number", "")
    if not re.match(r"^\d{4}\s?\d{6}$", series_number):
        errors.append(
            f"Неверный формат серии/номера паспорта: '{series_number}'. "
            f"Ожидается формат: XXXX XXXXXX"
        )

    birth_date_str = data.get("birth_date", "")
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        if birth_date > datetime.now():
            errors.append("Дата рождения не может быть в будущем")
        if birth_date.year < 1900:
            errors.append("Дата рождения не может быть раньше 1900 года")
    except (ValueError, TypeError):
        errors.append(
            f"Некорректный формат даты рождения: '{birth_date_str}'. "
            f"Ожидается формат: YYYY-MM-DD"
        )

    full_name = data.get("full_name", "")
    if not full_name:
        errors.append("ФИО не может быть пустым")
    elif not re.match(r"^[А-ЯЁа-яё\s\-]+$", full_name):
        errors.append(
            f"ФИО содержит недопустимые символы: '{full_name}'. "
            f"Допустимы только кириллица, пробелы и дефис"
        )

    return errors


def validate_contract_data(data: dict) -> list[str]:
    """Валидирует данные банковского договора."""
    errors = []

    if not data.get("contract_number"):
        errors.append("Номер договора не может быть пустым")

    amount = data.get("amount")
    if amount is not None:
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                errors.append(f"Сумма должна быть положительной: {amount}")
            if amount_float > 1_000_000_000:
                errors.append(f"Сумма подозрительно большая: {amount} — требует проверки")
        except (ValueError, TypeError):
            errors.append(f"Некорректный формат суммы: '{amount}'")

    rate = data.get("rate")
    if rate is not None:
        try:
            rate_float = float(rate)
            if not (0 <= rate_float <= 100):
                errors.append(f"Ставка должна быть от 0 до 100%: {rate}")
        except (ValueError, TypeError):
            errors.append(f"Некорректный формат ставки: '{rate}'")

    sign_date_str = data.get("sign_date", "")
    if sign_date_str:
        try:
            datetime.strptime(sign_date_str, "%Y-%m-%d")
        except ValueError:
            errors.append(
                f"Некорректный формат даты подписания: '{sign_date_str}'. "
                f"Ожидается формат: YYYY-MM-DD"
            )

    return errors


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
