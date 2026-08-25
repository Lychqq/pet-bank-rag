import unittest
from src.ocr.validators import validate_passport_data, validate_contract_data


class TestValidation(unittest.TestCase):
    def test_passport_validation_valid(self):
        valid_data = {
            "series_number": "4510 123456",
            "birth_date": "1995-05-15",
            "full_name": "Иванов Иван Иванович",
        }
        errors = validate_passport_data(valid_data)
        self.assertEqual(len(errors), 0)

    def test_passport_validation_invalid_series(self):
        invalid_data = {
            "series_number": "123",
            "birth_date": "1995-05-15",
            "full_name": "Иванов Иван",
        }
        errors = validate_passport_data(invalid_data)
        self.assertTrue(any("серии/номера" in e for e in errors))

    def test_passport_validation_invalid_name(self):
        invalid_data = {
            "series_number": "4510 123456",
            "birth_date": "1995-05-15",
            "full_name": "John Doe 123",
        }
        errors = validate_passport_data(invalid_data)
        self.assertTrue(any("недопустимые символы" in e for e in errors))

    def test_contract_validation_valid(self):
        valid_data = {
            "contract_number": "КР-2024-001",
            "amount": 500000.0,
            "rate": 14.5,
            "sign_date": "2024-01-10",
        }
        errors = validate_contract_data(valid_data)
        self.assertEqual(len(errors), 0)

    def test_contract_validation_negative_amount(self):
        invalid_data = {
            "contract_number": "КР-2024-001",
            "amount": -5000,
            "rate": 14.5,
        }
        errors = validate_contract_data(invalid_data)
        self.assertTrue(any("положительной" in e for e in errors))

    def test_contract_validation_exorbitant_amount(self):
        invalid_data = {
            "contract_number": "КР-2024-001",
            "amount": 2_000_000_000,
            "rate": 14.5,
        }
        errors = validate_contract_data(invalid_data)
        self.assertTrue(any("подозрительно большая" in e for e in errors))

    def test_contract_validation_invalid_rate(self):
        invalid_data = {
            "contract_number": "КР-2024-001",
            "amount": 100000,
            "rate": 150.0,
        }
        errors = validate_contract_data(invalid_data)
        self.assertTrue(any("Ставка должна быть от 0 до 100%" in e for e in errors))

