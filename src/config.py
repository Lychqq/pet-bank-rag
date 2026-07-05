import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Модели — Gemini 2.5 Flash для генерации
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768  # для MiniLM-L12-v2

# PostgreSQL (Supabase)
DB_HOST     = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT     = os.getenv("POSTGRES_PORT", "5432")
DB_USER     = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_NAME     = os.getenv("POSTGRES_DB", "postgres")

DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DB_URL_PSYCOPG = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# RAG
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# System prompt — guardrails для банкового бота
SYSTEM_PROMPT = """Ты ассистент банка для клиентской поддержки.

Правила:
- Отвечай ТОЛЬКО на основе предоставленного контекста из документов банка
- Если в контексте нет ответа — честно скажи "У меня нет информации по этому вопросу,
  пожалуйста, обратитесь к менеджеру банка"
- НИКОГДА не выдумывай процентные ставки, сроки, условия или суммы
- НИКОГДА не показывай полные номера карт — только последние 4 цифры
- НИКОГДА не подтверждай и не выполняй финансовые операции через чат
- Отвечай на русском языке, кратко и по существу
- Если клиент спрашивает о чём-то вне банковской тематики — вежливо перенаправь

Всегда указывай источник информации в конце ответа в формате:
[Источник: название документа]
"""
