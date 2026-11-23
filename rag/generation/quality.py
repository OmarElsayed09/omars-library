"""Resolve reply language and check a completed draft before showing it."""
import re


def requested_language(question: str) -> str | None:
    text = question.translate(str.maketrans("أإآ", "ااا"))
    command = re.search(
        r"(?:\b(?:speak|talk|continue|answer|respond|reply|write)\s+in\s+|"
        r"(?:نتكلم|نكمل|كمل|اتكلم|تتكلم|رد|جاوب|اكتب)\s+(?:(?:معايا|معي|علي)\s+)?)"
        r"(english\b|arabic\b|(?:بال)?انجليزي(?:ة)?|(?:بال)?عربي(?:ة)?)",
        text, re.I,
    )
    if command:
        return "en" if command[1].lower() == "english" or "انجليزي" in command[1] else "ar"
    return None


def reply_language(question: str, preference: str, previous: str | None = None) -> str:
    if preference != "auto":
        return preference
    requested = requested_language(question)
    if requested or previous:
        return requested or previous
    if re.search(r"[\u0621-\u064a]", question):
        return "ar"
    return "en"


def casual_reply(question: str, language: str) -> str | None:
    """Keep short greetings deterministic instead of spending a model turn on them."""
    normalized = re.sub(r"[^a-z\u0621-\u064a]+", "", question.lower())
    arabic_greetings = {
        "ازيك", "ازيكك", "عامل ايه", "عاملايه", "اهلا", "اهلًا", "مرحبا",
        "السلامعليكم", "صباحالخير", "مساءالخير",
    }
    english_greetings = {"hi", "hello", "hey", "thanks", "thankyou"}
    if normalized in arabic_greetings | english_greetings and language == "ar":
        return "أهلًا! أنا كويس، شكرًا لسؤالك. إزاي أقدر أساعدك؟"
    if normalized in arabic_greetings | english_greetings:
        return "Hi! How can I help you?"
    return None


def quality_issue(answer: str, language: str, sources: int, question: str) -> str:
    plain = re.sub(r"```[\s\S]*?```|`[^`\n]+`", " ", answer)
    citations = re.findall(r"\[(\d+)\]", plain)
    if any(int(ref) not in range(1, sources + 1) for ref in citations) and sources:
        return "Use only evidence numbers provided."
    if sources and not citations:
        return "Ground the answer in the supplied excerpts and cite at least one evidence number."
    prose = re.sub(r"\[[^\]\n]+\]", " ", plain)
    if not any(c.isalnum() for c in prose) and not re.search(r"`[^\s`][\s\S]*?`", answer):
        return "Give a substantive answer, not just a citation."
    if re.search(r"[\u3400-\u9fff]", prose) and not re.search(r"Chinese|Mandarin|Japanese|صيني|ياباني|[\u3400-\u9fff]", question, re.I):
        return "Rewrite the entire answer in the requested language."
    if re.search(r"[\u0400-\u052f]", prose) and not re.search(r"Russian|Ukrainian|Cyrillic|روسي|أوكراني", question, re.I):
        return "Rewrite the entire answer in the requested language; do not use Cyrillic script."
    arabic = len(re.findall(r"[\u0621-\u064a]", prose))
    latin = len(re.findall("[a-z]", prose, re.I))
    if language == "ar" and latin > 18 and not arabic and not re.search(r"بالإنجليزي|بالانجليزي|in english", question, re.I):
        return "Explain in Arabic; preserve necessary technical terms."
    if language == "en" and arabic > max(8, latin) and not re.search("Arabic|عربي", question, re.I):
        return "Explain in English."
    return ""
