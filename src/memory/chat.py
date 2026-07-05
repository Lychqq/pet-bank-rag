"""
Память диалога — персистентное хранение истории через PostgreSQL (Supabase).

Используем RunnableWithMessageHistory.
История хранится в Postgres — переживает перезагрузки сервера.

Почему не InMemoryChatMessageHistory:
- InMemory исчезает при перезапуске сервера
- В банковском контексте это критично — клиент упомянул важную деталь,
  сервер перезагрузился при деплое, бот "забыл" — плохой UX и возможные риски

session_id изолирует историю разных клиентов друг от друга.
"""

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PostgresChatMessageHistory

from src.config import GOOGLE_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT, DB_URL_PSYCOPG


def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
    )


def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    """
    Фабрика истории — возвращает историю для конкретного session_id.

    Почему функция, а не один объект:
    - Один общий объект памяти = все клиенты видят чужие диалоги
    - Функция с session_id = каждый клиент изолирован в своей "комнате"
    """
    return PostgresChatMessageHistory(  # pylint: disable=unexpected-keyword-arg
        connection_string=DB_URL_PSYCOPG,
        session_id=session_id,
        table_name="chat_history",
    )


def build_chain_with_memory():
    """
    Строит цепочку LLM с персистентной памятью.

    RunnableWithMessageHistory — актуальный подход (не legacy ConversationChain).
    Обёртка вокруг обычной цепочки — сама цепочка не знает про память,
    память управляется снаружи.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),  # ← история подставляется сюда
        ("human", "{input}"),
    ])

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def chat_with_memory(
    user_message: str,
    session_id: str,
    context: str = "",
) -> str:
    """
    Отправляет сообщение боту с учётом истории диалога.

    context — релевантные документы из RAG (если есть).
    Комбинируем RAG-контекст с памятью диалога.
    """
    chain = build_chain_with_memory()

    # Если есть RAG контекст — добавляем к сообщению
    if context:
        full_input = f"Контекст из документов банка:\n{context}\n\nВопрос клиента: {user_message}"
    else:
        full_input = user_message

    response = chain.invoke(
        {"input": full_input},
        config={"configurable": {"session_id": session_id}},
    )

    return response.content


def get_history(session_id: str) -> list[dict]:
    """Возвращает историю диалога для отображения."""
    history = get_session_history(session_id)
    messages = history.messages

    return [
        {
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
        }
        for msg in messages
    ]


def clear_history(session_id: str) -> None:
    """Очищает историю диалога (например при начале нового разговора)."""
    history = get_session_history(session_id)
    history.clear()
