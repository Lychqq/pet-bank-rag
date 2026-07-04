"""
Банковский RAG-ассистент — главный файл запуска.

Демонстрирует работу всех компонентов системы:
1. Индексация документов в pgvector
2. Диалог с ботом (RAG + память)
3. Обработка документов (OCR + валидация)
4. Оценка качества (Ragas)

Запуск:
  python main.py --mode index       # индексировать документы из docs/
  python main.py --mode chat        # диалог с ботом
  python main.py --mode ocr FILE    # извлечь данные из документа
  python main.py --mode eval        # запустить Ragas оценку
  python main.py --mode demo        # полная демонстрация всех компонентов
"""

import argparse
import sys
import uuid


def cmd_index(docs_folder: str = "docs"):
    """Индексирует PDF документы из указанной папки."""
    from src.rag.pipeline import index_documents
    print(f"Индексирую документы из папки '{docs_folder}'...")
    count = index_documents(docs_folder)
    if count > 0:
        print(f"\n[OK] Успешно! Добавлено {count} чанков в pgvector.")
    else:
        print("\n[WARN] Документы не найдены. Добавьте PDF файлы в папку docs/")


def cmd_chat():
    """Интерактивный диалог с банковским ботом."""
    from src.rag.pipeline import rag_query
    from src.memory.chat import chat_with_memory, get_history
    session_id = str(uuid.uuid4())
    print("\n" + "="*60)
    print("БАНКОВСКИЙ АССИСТЕНТ")
    print("="*60)
    print(f"Сессия: {session_id[:8]}...")
    print("Введите 'exit' для выхода, 'history' для просмотра истории")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("До свидания!")
            break

        if user_input.lower() == "history":
            history = get_history(session_id)
            print(f"\nИстория диалога ({len(history)} сообщений):")
            for msg in history:
                role = "Вы" if msg["role"] == "user" else "Бот"
                print(f"  {role}: {msg['content'][:100]}...")
            print()
            continue

        # RAG retrieval
        rag_result = rag_query(query=user_input, session_id=session_id)

        # Формируем контекст для передачи в чат с памятью
        if rag_result["sources"]:
            context_str = f"Найдено в документах ({', '.join(set(rag_result['sources']))})"
        else:
            context_str = ""

        # Генерация ответа с памятью диалога
        if rag_result["num_docs_relevant"] > 0:
            answer = rag_result["answer"]
        else:
            # Если RAG не нашёл ничего — отвечаем через память без RAG контекста
            answer = chat_with_memory(user_input, session_id)

        print(f"\nБот: {answer}")

        # Показываем метаданные если был Corrective RAG
        if rag_result.get("rewritten_query"):
            print(f"\n[Corrective RAG: запрос переписан для улучшения поиска]")

        print()


def cmd_ocr(file_path: str):
    """Извлекает и валидирует данные из документа."""
    from src.ocr.extractor import process_document
    print(f"\nОбрабатываю документ: {file_path}")
    print("="*60)

    result = process_document(file_path)

    print(f"Статус:     {result['status']}")
    print(f"Тип:        {result.get('document_type', 'unknown')}")

    if result["status"] == "extraction_failed":
        print(f"Ошибка:     {result['error']}")
        return

    print("\nИзвлечённые данные:")
    data = result.get("data", {})
    for key, value in data.items():
        if key != "document_type":
            print(f"  {key}: {value}")

    if result["validation_errors"]:
        print(f"\n[WARN] Ошибки валидации ({len(result['validation_errors'])}):")
        for error in result["validation_errors"]:
            print(f"  - {error}")
        print("\nТребуется проверка менеджером!")
    else:
        print("\n[OK] Валидация прошла успешно!")


def cmd_eval():
    """Запускает Ragas оценку качества RAG пайплайна."""
    from src.evaluation.ragas_eval import run_evaluation
    print("\nЗапуск оценки качества RAG пайплайна через Ragas...")
    print("Это может занять несколько минут (LLM-as-judge).\n")
    results = run_evaluation()
    return results


def cmd_demo():
    """Полная демонстрация всех компонентов системы."""
    from src.rag.pipeline import rag_query
    from src.memory.chat import chat_with_memory
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ БАНКОВСКОГО RAG-АССИСТЕНТА")
    print("="*60)

    session_id = "demo_session_001"

    # 1. Показываем RAG запрос с Corrective RAG
    print("\n--- 1. RAG ЗАПРОС (с Corrective RAG) ---\n")
    test_questions = [
        "Какая процентная ставка по ипотеке?",
        "Как заблокировать украденную карту?",
        "Какой максимальный кредитный лимит по золотой карте?",  # нет в документах
    ]

    for q in test_questions:
        print(f"Вопрос: {q}")
        result = rag_query(query=q, session_id=session_id)
        print(f"Ответ: {result['answer'][:200]}...")
        print(f"  Grade: {result['grade']} | "
              f"Docs: {result['num_docs_retrieved']} найдено, "
              f"{result['num_docs_relevant']} релевантных")
        if result.get("rewritten_query"):
            print(f"  [Corrective RAG активирован] Переписан: {result['rewritten_query']}")
        print()

    # 2. Показываем память диалога
    print("\n--- 2. ПАМЯТЬ ДИАЛОГА ---\n")
    demo_session = "demo_memory_001"
    messages = [
        "Меня зовут Алексей, я хочу взять ипотеку",
        "Как меня зовут и о чём я спрашивал?",  # проверяем память
    ]
    for msg in messages:
        print(f"Клиент: {msg}")
        response = chat_with_memory(msg, demo_session)
        print(f"Бот: {response[:200]}...")
        print()

    print("\n[OK] Демонстрация завершена!")
    print("\nДля запуска оценки качества: python main.py --mode eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Банковский RAG-ассистент")
    parser.add_argument(
        "--mode",
        choices=["index", "chat", "ocr", "eval", "demo"],
        required=True,
        help="Режим запуска",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Путь к файлу (для режима ocr)",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="docs",
        help="Папка с PDF документами (для режима index)",
    )

    args = parser.parse_args()

    if args.mode == "index":
        cmd_index(args.docs)
    elif args.mode == "chat":
        cmd_chat()
    elif args.mode == "ocr":
        if not args.file:
            print("Ошибка: укажите файл через --file")
            sys.exit(1)
        cmd_ocr(args.file)
    elif args.mode == "eval":
        cmd_eval()
    elif args.mode == "demo":
        cmd_demo()
