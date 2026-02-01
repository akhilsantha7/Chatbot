# guardrails.py

from typing import List, Optional

def is_allowed_question(question: str, memory: Optional[List[str]] = None) -> bool:


    text = question.lower()

    # Keywords allowed for avocado-related queries
    allowed_keywords = [
        "avocado", "persea",
        "irrigation", "fertilizer", "disease",
        "temperature", "spacing", "yield",
        "root rot", "anthracnose", "sun blotch",
        "cercospora", "powdered mildew"
    ]

    # Basic keyword check
    if any(word in text for word in allowed_keywords):
        return True

    # Follow-up check: if last user question is about avocado, allow follow-up
    if memory and len(memory) >= 2:
        last_q = memory[-2].lower()  # memory is alternating Q/A
        if "avocado" in last_q:
            return True

    # Greetings or exit keywords
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
    exits = ["bye", "thank you", "thanks", "good night"]

    if any(greet in text for greet in greetings):
        return True
    if any(exit_word in text for exit_word in exits):
        return True

    # Otherwise, disallow
    return False
