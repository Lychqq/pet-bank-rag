"""
Детерминированная валидация данных документов (паспорта, договоры).
Чистый Python без внешних зависимостей.
"""

import re
from datetime import datetime


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
