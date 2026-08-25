import logging
import time
import psycopg2
from src.config import DB_URL_PSYCOPG, DB_URL

logger = logging.getLogger(__name__)

def get_connection():
    """Возвращает соединение к БД с защитой от обрывов Supabase."""
    for attempt in range(5):
        try:
            conn = psycopg2.connect(
                DB_URL_PSYCOPG,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            conn.autocommit = True
            return conn
        except psycopg2.Error as e:
            logger.warning("[БД] Ошибка подключения: %s. Попытка %d/5, повтор через 2 сек...", e, attempt + 1)
            time.sleep(2)
    raise ConnectionError("Не удалось подключиться к базе данных после 5 попыток.")


def get_sync_connection_string() -> str:
    """Строка подключения для LangChain-postgres компонентов."""
    return DB_URL
