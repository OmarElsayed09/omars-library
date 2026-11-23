import math
import re
import unicodedata
from collections import Counter

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}
ARABIC_STOPWORDS = {
    "ما", "ماذا", "متى", "اين", "كيف", "هل", "هو", "هي", "في", "من",
    "على", "عن", "الى", "و", "او", "هذا", "هذه", "ذلك", "تلك", "التي",
    "الذي", "ثم", "كان", "كانت", "يكون",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ـ", "")
    normalized = "".join(
        char for char in normalized
        if not unicodedata.category(char).startswith("M")
    )
    normalized = normalized.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ة": "ه", "ى": "ي",
    }))
    terms = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    normalized_terms = []
    for term in terms:
        if term in ENGLISH_STOPWORDS or term in ARABIC_STOPWORDS:
            continue
        if len(term) > 4 and term.endswith("ies"):
            term = term[:-3] + "y"
        elif len(term) > 5 and term.endswith(("sses", "shes", "ches", "xes", "zes")):
            term = term[:-2]
        elif len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
            term = term[:-1]
        normalized_terms.append(term)
    return normalized_terms


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: dict[str, str] = {}
        self.metadata: dict[str, dict] = {}
        self.term_counts: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_frequency: Counter[str] = Counter()
        self.average_length = 0.0

    def rebuild(self, documents: dict[str, str], metadata: dict[str, dict] | None = None) -> None:
        self.documents = dict(documents)
        self.metadata = dict(metadata or {})
        self.term_counts = {}
        self.doc_lengths = {}
        self.doc_frequency = Counter()
        for doc_id, text in documents.items():
            terms = tokenize(text)
            counts = Counter(terms)
            self.term_counts[doc_id] = counts
            self.doc_lengths[doc_id] = len(terms)
            self.doc_frequency.update(counts.keys())
        self.average_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths else 0.0
        )

    def search(self, query: str, limit: int = 10, source: str | None = None) -> list[dict]:
        if not query.strip() or not self.documents:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        scored = []
        count = len(self.documents)
        for doc_id, text in self.documents.items():
            if source is not None and self.metadata.get(doc_id, {}).get("source") != source:
                continue
            length = self.doc_lengths[doc_id]
            score = 0.0
            for term in query_terms:
                frequency = self.term_counts[doc_id].get(term, 0)
                if not frequency:
                    continue
                df = self.doc_frequency.get(term, 0)
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                normalization = self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1.0)
                )
                score += idf * frequency * (self.k1 + 1) / (frequency + normalization)
            if score > 0:
                scored.append((score, doc_id, text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": doc_id,
                "content": text,
                "score": score,
                "metadata": self.metadata.get(doc_id, {}),
            }
            for score, doc_id, text in scored[:max(limit, 0)]
        ]
