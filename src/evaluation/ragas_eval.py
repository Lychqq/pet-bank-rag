"""
Оценка качества RAG через Ragas.

Раздельная оценка retrieval и generation:

RETRIEVAL:
  context_precision — из найденного, сколько реально релевантно вопросу?
  context_recall    — нашли ли всё что нужно для полного ответа?

GENERATION:
  faithfulness      — ответ основан на контексте, или модель выдумала?
  answer_relevancy  — ответ релевантен вопросу?

LLM-as-judge: Ragas использует LLM для оценки (Gemini в нашем случае).
Слабость: та же модель судит свои же ответы. Поэтому используем как
инструмент ИТЕРАЦИИ при разработке, а не как абсолютную истину.
"""

import sys
import types
if 'langchain_community.chat_models' not in sys.modules:
    sys.modules['langchain_community.chat_models'] = types.ModuleType('chat_models')
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    sys.modules['langchain_community.chat_models.vertexai'] = types.ModuleType('vertexai')
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = type('ChatVertexAI', (object,), {})
from datasets import Dataset
from ragas import evaluate
# pylint: disable=no-name-in-module
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
# pylint: enable=no-name-in-module
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, GEMINI_MODEL
from src.rag.pipeline import rag_query

def get_ragas_llm():
    """LLM для судьи Ragas (возвращаем Google Gemini, так как Groq не тянет лимиты)."""
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )
    )


from ragas.embeddings.base import BaseRagasEmbeddings
from src.rag.pipeline import get_embeddings

class CustomRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self):
        super().__init__()
        self.model = get_embeddings()
    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)
    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

def get_ragas_embeddings():
    """Embeddings для Ragas (через наш локальный SentenceTransformer)."""
    return CustomRagasEmbeddings()


# =============================================
# Golden dataset — тестовые вопросы с эталонными ответами
# =============================================

GOLDEN_DATASET = [
    # Вопросы по Альфа-Карте (alfacard.pdf)
    {"question": "Сколько стоит обслуживание дебетовой Альфа-Карты?", "ground_truth": "Обслуживание Альфа-Карты полностью бесплатно без дополнительных условий."},
    {"question": "Какая комиссия за снятие наличных с Альфа-Карты в банкоматах других банков?", "ground_truth": "В банкоматах других банков можно снимать до 100 000 рублей в месяц бесплатно. При снятии свыше 100 000 рублей в месяц комиссия составит 1,99%, минимум 199 рублей."},
    {"question": "Какая стоимость уведомлений об операциях по Альфа-Карте?", "ground_truth": "Услуга уведомления об операциях по Альфа-Карте стоит 99 рублей."},

    # Вопросы по Детской карте (tariff-kids-card.pdf)
    {"question": "Для детей какого возраста предназначена Детская карта Альфа-Банка?", "ground_truth": "Детская карта предназначена для детей в возрасте от 6 до 14 лет."},
    {"question": "Сколько стоит перевыпуск утраченной Детской карты?", "ground_truth": "Перевыпуск утраченной Детской карты осуществляется бесплатно."},
    {"question": "Какой максимальный размер кэшбэка можно получить по Детской карте в месяц?", "ground_truth": "Максимальный размер кэшбэка по Детской карте составляет 2 000 рублей в месяц."},

    # Вопросы по ипотечным тарифам (mortgage_tariffs.pdf)
    {"question": "Какая комиссия взимается за выдачу выписки (дубликата выписки) по ипотечному счету?", "ground_truth": "За выдачу выписки или дубликата выписки по ипотечному счету взимается комиссия в размере 150 рублей за каждую выписку."},
    {"question": "Сколько стоит открытие дополнительного счета для зачисления средств материнского капитала по ипотеке?", "ground_truth": "За открытие дополнительного счета для зачисления средств материнского (семейного) капитала комиссия не установлена (бесплатно)."},
    {"question": "Сколько стоит открытие аккредитива в валюте РФ по ипотечным сделкам?", "ground_truth": "Комиссия за открытие аккредитива в валюте РФ составляет 3500 рублей."},
    {"question": "Сколько стоит заказать справку об исполнении обязательств по ипотечному договору через А-Клик или А-Мобайл?", "ground_truth": "При запросе через удаленные каналы доступа (А-Клик и А-Мобайл) справка об исполнении обязательств предоставляется бесплатно (комиссия не установлена)."}
]


def run_evaluation() -> dict:
    """
    Запускает полную оценку RAG пайплайна через Ragas.

    Для каждого вопроса из golden dataset:
    1. Прогоняем через наш RAG пайплайн
    2. Собираем вопрос, ответ, контекст, эталонный ответ
    3. Ragas считает метрики через LLM-as-judge

    Возвращает словарь с метриками и детальными результатами.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    grades = []
    rewritten_queries = []

    print("Запуск оценки RAG пайплайна...")
    print(f"Тестовых вопросов: {len(GOLDEN_DATASET)}\n")

    from src.database import get_connection
    conn = get_connection()
    try:
        for i, item in enumerate(GOLDEN_DATASET):
            print(f"[{i+1}/{len(GOLDEN_DATASET)}] {item['question'][:60]}...")

            result = rag_query(
                query=item["question"],
                session_id=f"ragas_eval_{i}",
                conn=conn,
            )

            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append([
                f"[{src}]" for src in result["sources"]
            ] if result["sources"] else ["Контекст не найден"])
            ground_truths.append(item["ground_truth"])
            grades.append(result["grade"])
            rewritten_queries.append(result.get("rewritten_query"))

            print(f"  Grade: {result['grade']} | "
                  f"Docs retrieved: {result['num_docs_retrieved']} | "
                  f"Relevant: {result['num_docs_relevant']}")
            if result.get("rewritten_query"):
                print(f"  Запрос переписан: {result['rewritten_query'][:60]}...")
    finally:
        if not conn.closed:
            conn.close()

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1, max_wait=60, max_retries=10)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nЗапуск Ragas метрик (LLM-as-judge)...")

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=run_config,
    )

    scores = result.to_pandas()
    summary = {
        "context_precision":  round(float(scores["context_precision"].mean()), 3),
        "context_recall":     round(float(scores["context_recall"].mean()), 3),
        "faithfulness":       round(float(scores["faithfulness"].mean()), 3),
        "answer_relevancy":   round(float(scores["answer_relevancy"].mean()), 3),
    }

    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ RAG ПАЙПЛАЙНА")
    print("="*50)
    for metric, score in summary.items():
        status = "[OK]" if score >= 0.7 else "[WARN]" if score >= 0.5 else "[FAIL]"
        print(f"{status} {metric:25s}: {score:.3f}")

    print("\nCorrective RAG статистика:")
    rewritten_count = sum(1 for q in rewritten_queries if q is not None)
    print(f"  Запросов переписано: {rewritten_count}/{len(GOLDEN_DATASET)}")
    not_relevant_count = sum(1 for g in grades if g == "not_relevant")
    print(f"  Нерелевантных retrievals: {not_relevant_count}/{len(GOLDEN_DATASET)}")

    return {
        "summary": summary,
        "detailed": scores.to_dict(orient="records"),
        "corrective_rag_stats": {
            "total_questions": len(GOLDEN_DATASET),
            "queries_rewritten": rewritten_count,
            "not_relevant_retrievals": not_relevant_count,
        }
    }
