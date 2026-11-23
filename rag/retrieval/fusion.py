from collections.abc import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict]],
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = str(item.get("id") or item.get("document_id") or item.get("content", "")[:256])
            if doc_id not in fused:
                fused[doc_id] = dict(item)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [{**fused[doc_id], "rrf_score": scores[doc_id]} for doc_id in ranked[:limit]]
