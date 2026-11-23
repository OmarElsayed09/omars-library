import os
import time


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "RERANK_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )
        self.model = None
        self.available = False

    def load(self) -> bool:
        if self.model is not None:
            return self.available
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device="cpu")
            self.available = True
        except Exception:
            self.available = False
        return self.available

    def rerank(self, query: str, records: list[dict], limit: int = 4) -> tuple[list[dict], float]:
        started = time.perf_counter()
        if not records:
            return [], 0.0
        if not self.load():
            elapsed = (time.perf_counter() - started) * 1000
            return records[:limit], elapsed
        scores = self.model.predict(
            [(query, str(record.get("content", ""))) for record in records],
            show_progress_bar=False,
        )
        reranked = []
        for record, score in zip(records, scores):
            reranked.append({**record, "rerank_score": float(score)})
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        elapsed = (time.perf_counter() - started) * 1000
        return reranked[:limit], elapsed
