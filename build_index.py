import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

VECTOR_DB_PATH = "vector_db"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# 1. Load documents
loader = PyPDFLoader("data/Avocado_Farming_Guide_Telangana_Academic.pdf")
docs = loader.load()

# 2. Embeddings
embeddings = OpenAIEmbeddings()

# 3. Build FAISS
db = FAISS.from_documents(docs, embeddings)

# 4. Save index
db.save_local(VECTOR_DB_PATH)

print("✅ FAISS index created successfully")
