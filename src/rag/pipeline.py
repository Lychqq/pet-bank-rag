import json
import math
import re
import time
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    SYSTEM_PROMPT,
)
from src.database import get_connection

class LocalEmbeddings:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_documents(self, texts):
        return [emb.tolist() for emb in self.model.encode(texts)]
        
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

_embeddings_instance = None
def get_embeddings():
    global _embeddings_instance  # pylint: disable=global-statement
    if _embeddings_instance is None:
        _embeddings_instance = LocalEmbeddings()
    return _embeddings_instance



def get_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )

def index_documents(docs_folder: str = "docs") -> int:
    embeddings_model = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    total_chunks = 0
    pdf_files = list(Path(docs_folder).glob("*.pdf"))
    if not pdf_files:
        return 0

    global_conn = get_connection()
    
    for pdf_path in pdf_files:
        print(f"Индексирую: {pdf_path.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            chunks = splitter.split_documents(loader.load())
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  [ERROR] Ошибка чтения PDF: {e}")
            continue

        if not chunks:
            continue

        texts = [chunk.page_content for chunk in chunks]
        
        try:
            batch_embs = embeddings_model.model.encode(texts, batch_size=256, show_progress_bar=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  [ERROR] Ошибка генерации эмбеддингов: {e}")
            continue
        
        params = []
        for chunk, emb in zip(chunks, batch_embs):
            if any(math.isnan(x) for x in emb):
                continue
            clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', chunk.page_content)
            params.append((
                clean_text,
                json.dumps({"source": pdf_path.name, "page": chunk.metadata.get("page", 0)}),
                "[" + ",".join(str(x) for x in emb) + "]",
            ))
        
        inserted_for_pdf = 0
        batch_size_db = 100
        
        if params:
            for i in range(0, len(params), batch_size_db):
                batch = params[i:i + batch_size_db]
                for attempt in range(5):
                    try:
                        if global_conn.closed:
                            global_conn = get_connection()
                        cur = global_conn.cursor()
                        query = "INSERT INTO documents (content, metadata, embedding) VALUES %s ON CONFLICT DO NOTHING"
                        psycopg2.extras.execute_values(cur, query, batch)
                        cur.close()
                        inserted_for_pdf += len(batch)
                        break
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"Ошибка вставки, попытка {attempt+1}: {e}")
                        time.sleep(1)
                        if not global_conn.closed:
                            try: global_conn.close()
                            except Exception: pass  # pylint: disable=broad-exception-caught
                        if attempt < 4:
                            try: global_conn = get_connection()
                            except Exception: pass  # pylint: disable=broad-exception-caught
                            
        total_chunks += inserted_for_pdf
        print(f"  OK {pdf_path.name}: {inserted_for_pdf} chunks")

    if not global_conn.closed: global_conn.close()
    return total_chunks

def hybrid_search(query: str, conn, k: int = TOP_K_RETRIEVAL) -> list[dict]:
    query_embedding = get_embeddings().embed_query(query)
    cur = conn.cursor()
    cur.execute(
        """
        WITH vector_search AS (
            SELECT id, content, metadata, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
            FROM documents LIMIT %s
        ),
        bm25_search AS (
            SELECT id, content, metadata, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(ts, plainto_tsquery('russian', %s)) DESC) AS rank
            FROM documents WHERE ts @@ plainto_tsquery('russian', %s) LIMIT %s
        ),
        rrf AS (
            SELECT COALESCE(v.id, b.id) AS id, COALESCE(v.content, b.content) AS content, COALESCE(v.metadata, b.metadata) AS metadata,
                   COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + b.rank), 0) AS score
            FROM vector_search v FULL OUTER JOIN bm25_search b USING (id)
        )
        SELECT id, content, metadata, score FROM rrf ORDER BY score DESC LIMIT %s
        """,
        ("[" + ",".join(str(x) for x in query_embedding) + "]", k * 2, query, query, k * 2, k)
    )
    results = cur.fetchall()
    cur.close()
    return [{"id": r[0], "content": r[1], "metadata": r[2], "score": float(r[3])} for r in results]

def grade_documents(query: str, documents: list[dict]) -> tuple[str, list[dict]]:
    if not documents:
        return "not_relevant", []
        
    llm = get_llm(temperature=0.0)
    context = "\n\n".join([d['content'] for d in documents])
    prompt = f"""Ты — строгий оценщик (Grader). 
Оцени, содержит ли предложенный контекст факты, необходимые для ответа на вопрос пользователя.
Тебе не нужно отвечать на сам вопрос, только сказать, полезен ли контекст.
Ответь ТОЛЬКО одним словом: 'yes' (если релевантно) или 'no' (если нет).

Вопрос: {query}
Контекст:
{context}"""

    response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    
    if "yes" in response or "да" in response:
        return "relevant", documents
    else:
        return "not_relevant", []

def rewrite_query(query: str) -> str:
    llm = get_llm(temperature=0.3)
    prompt = f"Перефразируй вопрос пользователя для улучшения поиска. Оригинальный вопрос: {query}\nВозврати ТОЛЬКО переформулированный вопрос."
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

def generate_answer(query: str, documents: list[dict]) -> str:
    llm = get_llm(temperature=0.0)
    context = "\n\n".join([f"[{d['metadata'].get('source', 'doc')}]\n{d['content']}" for d in documents])
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f"Контекст:\n{context}\n\nВопрос: {query}")]
    return llm.invoke(messages).content

def rag_query(query: str, session_id: Optional[str] = None, conn = None) -> dict:
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    docs = hybrid_search(query, conn)
    grade, relevant_docs = grade_documents(query, docs)

    if grade == "not_relevant" and docs:
        rewritten = rewrite_query(query)
        docs = hybrid_search(rewritten, conn)
        grade, relevant_docs = grade_documents(rewritten, docs)
    else:
        rewritten = None

    if relevant_docs:
        answer = generate_answer(query, relevant_docs)
    else:
        answer = "К сожалению, информации не найдено."

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO rag_logs (session_id, question, original_query, rewritten_query, grade, retrieved_docs, final_answer) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (session_id, query, query, rewritten, grade, json.dumps([{"content": d["content"][:200], "source": d["metadata"].get("source")} for d in relevant_docs]), answer)
        )
        cur.close()
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"\n[Внимание] Не удалось сохранить лог в БД (соединение разорвано): {e}")

    if close_conn: conn.close()
    return {"answer": answer, "grade": grade, "rewritten_query": rewritten, "sources": [d["metadata"].get("source") for d in relevant_docs], "num_docs_retrieved": len(docs), "num_docs_relevant": len(relevant_docs)}
