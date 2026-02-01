# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from intents import handle_question
from retriever import load_vectorstore  # import your loader

# ------------------------
# Models
# ------------------------

load_vectorstore() 

class ChatRequest(BaseModel):
    question: str
    memory: Optional[List[str]] = []

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    memory: List[str]

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(
    title="Telangana Avocado RAG Chatbot",
    description="RAG-based chatbot for avocado farming in Telangana, India.",
    version="1.0.0"
)

# Enable CORS for iOS / Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your iOS app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Routes
# ------------------------
@app.get("/")
async def root():
    return {"message": "Telangana Avocado RAG Chatbot is running ✅"}

@app.post("/ask", response_model=ChatResponse)
async def ask_chatbot(request: ChatRequest):
    """
    Receive a user question, return RAG answer and updated memory.
    """
    try:
        response = await handle_question(request.question, request.memory)
        return ChatResponse(**response)
    except Exception as e:
        return ChatResponse(
            answer=f"⚠️ Something went wrong: {str(e)}",
            confidence=0.0,
            memory=request.memory or []
        )
