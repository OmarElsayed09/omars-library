import math


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / k


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0
