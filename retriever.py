# retriever.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
VECTORSTORE = None

def load_vectorstore():
    """
    Load the FAISS vectorstore from local storage.
    This function caches the loaded vectorstore to avoid repeated I/O.
    """
    global VECTORSTORE
    if VECTORSTORE is None:
        try:
            VECTORSTORE = FAISS.load_local(
                "vector_db",  # path to your FAISS folder
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS vectorstore loaded successfully.")
        except Exception as e:
            print("❌ Error loading FAISS vectorstore:", e)
            raise e
    return VECTORSTORE

def retrieve_context(question: str, k: int = 3):
    """
    Retrieve the top-k relevant document chunks from FAISS based on the question.
    Returns a list of text strings (page_content).
    """
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]
