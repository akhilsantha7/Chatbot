# intents.py

from typing import List, Optional
from guardrails import is_allowed_question
from rag_core import generate_answer
import asyncio

# Greetings and exit responses
GREETINGS_RESPONSES = [
    "Hello! 👋 I can help with avocado farming in Telangana. Ask me about irrigation, diseases, spacing, yield, or temperature.",
    "Hi there! I provide evidence-based guidance on avocado cultivation in Telangana. What would you like to know?"
]

EXIT_RESPONSES = [
    "Goodbye! 🌱 Happy farming.",
    "Thanks for asking! Feel free to come back with more avocado questions."
]

DEFAULT_FALLBACK = "⚠️ Sorry, I can only answer avocado farming questions specific to Telangana."

async def handle_question(question: str, memory: Optional[List[str]] = None) -> dict:
    """
    Process a user's question, check guardrails, and generate an answer.

    Args:
        question (str): User's question.
        memory (List[str], optional): Chat memory (alternating Q/A).

    Returns:
        dict: {
            "answer": str,
            "confidence": float,
            "memory": List[str]
        }
    """
    memory = memory or []

    # Normalize
    text = question.lower()

    # 1️⃣ Check greetings
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
    if any(greet in text for greet in greetings):
        answer = GREETINGS_RESPONSES[len(memory) % len(GREETINGS_RESPONSES)]
        memory.append(f"Q: {question}")
        memory.append(f"A: {answer}")
        return {"answer": answer, "confidence": 1.0, "memory": memory}

    # 2️⃣ Check exit phrases
    exits = ["bye", "thank you", "thanks", "good night"]
    if any(exit_word in text for exit_word in exits):
        answer = EXIT_RESPONSES[len(memory) % len(EXIT_RESPONSES)]
        memory.append(f"Q: {question}")
        memory.append(f"A: {answer}")
        return {"answer": answer, "confidence": 1.0, "memory": memory}

    # 3️⃣ Guardrails check
    if not is_allowed_question(question, memory):
        memory.append(f"Q: {question}")
        memory.append(f"A: {DEFAULT_FALLBACK}")
        return {"answer": DEFAULT_FALLBACK, "confidence": 0.0, "memory": memory}

    # 4️⃣ Generate RAG-based answer
    try:
        answer_text = await generate_answer(question)
        # Confidence heuristic: fraction of words appearing in retrieved context
        context_words = " ".join([entry.lower() for entry in memory])
        match_count = sum(word in context_words for word in answer_text.lower().split())
        confidence = round(min(0.95, 0.4 + match_count * 0.05), 2)

        memory.append(f"Q: {question}")
        memory.append(f"A: {answer_text}")

        return {"answer": answer_text, "confidence": confidence, "memory": memory}
    except Exception as e:
        fallback_msg = f"⚠️ Something went wrong: {str(e)}"
        memory.append(f"Q: {question}")
        memory.append(f"A: {fallback_msg}")
        return {"answer": fallback_msg, "confidence": 0.0, "memory": memory}
