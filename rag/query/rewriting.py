import re
from ollama import RequestError, ResponseError


def english_search_query(query: str, sample: str, client, model: str) -> str:
    """Bridge Arabic questions to English lexical terms only for English passages."""
    latin_letters = len(re.findall("[A-Za-z]", sample))
    arabic_letters = len(re.findall(r"[\u0600-\u06ff]", sample))
    if not re.search(r"[\u0600-\u06ff]", query) or latin_letters < 3 * max(1, arabic_letters):
        return query
    try:
        result = client.chat(model=model, messages=[
            {"role": "system", "content": "Translate the question into a concise English search query. Preserve technical terms, numbers and proper nouns. Return only the question, never an answer."},
            {"role": "user", "content": query},
        ], options={"temperature": 0, "num_predict": 72})
        translated = (result.message.content or "").strip()
        return translated if re.search("[A-Za-z]", translated) else query
    except (OSError, RuntimeError, RequestError, ResponseError):
        return query


def needs_rewrite(query: str) -> bool:
    text = query.strip()
    if not text:
        return False
    words = set(re.findall(r"\w+", text.casefold()))
    markers = {"it", "this", "that", "those", "previous", "he", "she", "ده", "دي", "ذلك", "هذا", "هذه",
               "السابق", "كمل", "وضح", "وضحها", "اشرحها", "مثال", "أبسط", "ابسط", "عليه", "عليها"}
    return len(text.split()) <= 12 and bool(words & markers)


def rewrite_retrieval_query(
    query: str,
    ollama_client,
    model: str,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    """Best-effort, timeout-bounded query rewrite with a safe original-query fallback."""
    if not needs_rewrite(query):
        return query
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the latest user question into one retrieval-friendly question. "
                        "Use only information explicitly present in the question and provided "
                        "conversation. Never invent facts. Return only the rewritten question."
                    ),
                },
                *[
                    {"role": message["role"], "content": message["content"]}
                    for message in (conversation or [])
                    if message.get("role") in {"user", "assistant"}
                ],
                {"role": "user", "content": query},
            ],
            options={"num_predict": 96},
        )
        rewritten = str(response["message"].get("content") or "").strip()
        return rewritten if rewritten else query
    except Exception:
        return query
