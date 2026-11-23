from rag.retrieval.bm25 import BM25Index
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.reranker import CrossEncoderReranker


def dense_to_records(documents: list[str], metadatas: list[dict], distances: list[float], ids: list[str] | None = None) -> list[dict]:
    records = []
    for index, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        doc_id = ids[index] if ids and index < len(ids) else f"{metadata.get('source', 'unknown')}:{metadata.get('chunk', index)}"
        records.append({
            "id": doc_id,
            "document_id": doc_id,
            "content": document,
            "metadata": metadata,
            "distance": float(distance or 0.0),
        })
    return records


class HybridRetriever:
    def __init__(self, bm25: BM25Index, reranker: CrossEncoderReranker | None = None) -> None:
        self.bm25 = bm25
        self.reranker = reranker
        self.last_rerank_latency_ms = 0.0
        self.last_reranker_available = False

    def fuse(
        self,
        dense_records: list[dict],
        query: str,
        limit: int = 10,
        rrf_k: int = 60,
        lexical_records: list[dict] | None = None,
    ) -> list[dict]:
        if lexical_records is None:
            lexical_records = self.bm25.search(query, limit=max(limit * 3, limit))
        fused = reciprocal_rank_fusion(
            [dense_records[:max(limit * 3, limit)], lexical_records],
            k=rrf_k,
            limit=max(limit * 3, limit),
        )
        if self.reranker is not None:
            fused, elapsed = self.reranker.rerank(query, fused, limit=limit)
            self.last_rerank_latency_ms = elapsed
            self.last_reranker_available = self.reranker.available
        else:
            fused = fused[:limit]
        return fused
