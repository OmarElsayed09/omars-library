import re


def route_query(query: str) -> str:
    text = query.strip().lower()
    if not text:
        return "general"
    if re.search(r"\b(?:document|pdf|lecture|chapter|according to|(?:my|this|the|uploaded) file)\b", text) or any(
        marker in text for marker in ("المستند", "الملف", "المحاضرة", "الصفحة", "في المستند", "المصدر", "من ملف")
    ):
        return "document"
    follow_up_markers = (
        "it", "this", "that", "those", "he", "she", "they",
        "ده", "دي", "هذا", "هذه", "ذلك", "تلك", "السابق", "اللي قبله", "اشرحها", "عليه", "عليها", "كمل", "أبسط", "ابسط",
    )
    if any(marker == text or f" {marker} " in f" {text} " for marker in follow_up_markers):
        return "follow_up"
    casual = {"hi", "hello", "hey", "thanks", "thank you", "مرحبا", "السلام", "شكرا"}
    if text in casual:
        return "general"
    return "general"
