# rag_core.py
import asyncio
from typing import List
from retriever import retrieve_context
from langchain_openai import ChatOpenAI

# ⚠️ gpt-5-mini only supports default temperature=1
llm = ChatOpenAI(model="gpt-5-mini")  # temperature is default 1

async def generate_answer(question: str, top_k: int = 3) -> str:
    """
    Generate an answer using RAG + LLM for Telangana avocado queries.

    Args:
        question (str): The user question.
        top_k (int): Number of context chunks to retrieve from FAISS.

    Returns:
        str: LLM-generated answer.
    """
    # Retrieve top-k context chunks
    context_chunks: List[str] = retrieve_context(question, k=top_k)

    # Flatten context into a single string for prompt
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are an agricultural scientist specializing in avocado cultivation
in Telangana, India.

Use concise, academic FAO-style language.
Do not mention sections, pages, or sources.
Answer strictly based on the provided context.

Context:
{context_text}

Question:
{question}
"""

    loop = asyncio.get_event_loop()
    # Run blocking LLM call in executor
    response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
    return getattr(response, "content", str(response))
