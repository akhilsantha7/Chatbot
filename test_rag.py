# test_rag.py
import asyncio
from rag_core import generate_answer
from retriever import VECTORSTORE

def test_faiss():
    try:
        # Test FAISS retrieval
        docs = VECTORSTORE.similarity_search("Optimal temperature for avocado in Telangana?", k=3)
        print("FAISS retrieval successful. Top 3 documents:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:200]}...")  # print first 200 chars
    except Exception as e:
        print("❌ FAISS retrieval failed:", e)

async def test_llm():
    try:
        question = "Optimal temperature for avocado in Telangana?"
        answer = await generate_answer(question)
        print("\n✅ LLM answer:")
        print(answer)
    except Exception as e:
        print("❌ LLM call failed:", e)

if __name__ == "__main__":
    test_faiss()
    asyncio.run(test_llm())
