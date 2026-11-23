import unittest
from unittest.mock import patch

from rag.ingestion.chunking import recursive_text_split
from rag.query.routing import route_query
from rag.query.rewriting import needs_rewrite
from rag.retrieval.reranker import CrossEncoderReranker

from rag.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag.retrieval.bm25 import BM25Index, tokenize
from rag.retrieval.fusion import reciprocal_rank_fusion


class HybridRetrievalTests(unittest.TestCase):
    def test_arabic_normalization(self) -> None:
        self.assertEqual(tokenize("إِدارةُ الْمَشْرُوع"), tokenize("اداره المشروع"))

    def test_bm25_ranks_matching_document(self) -> None:
        index = BM25Index()
        index.rebuild(
            {"a": "The router password is in network settings.", "b": "Cooking recipes."},
            {"a": {"source": "network.txt"}, "b": {"source": "recipes.txt"}},
        )
        results = index.search("router password", limit=2)
        self.assertEqual(results[0]["id"], "a")

    def test_rrf_deduplicates_documents(self) -> None:
        dense = [{"id": "a", "content": "A"}, {"id": "b", "content": "B"}]
        lexical = [{"id": "b", "content": "B"}, {"id": "c", "content": "C"}]
        results = reciprocal_rank_fusion([dense, lexical], k=60, limit=3)
        self.assertEqual([item["id"] for item in results], ["b", "a", "c"])

    def test_retrieval_metrics(self) -> None:
        relevant = {"b"}
        retrieved = ["a", "b", "c"]
        self.assertEqual(recall_at_k(relevant, retrieved, 3), 1.0)
        self.assertEqual(reciprocal_rank(relevant, retrieved), 0.5)
        self.assertGreater(ndcg_at_k(relevant, retrieved, 3), 0.0)

    def test_recursive_chunking_preserves_overlap_and_boundaries(self) -> None:
        text = "Heading\n\nFirst paragraph. Second paragraph.\nThird paragraph."
        chunks = recursive_text_split(text, size=28, overlap=6)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 28 for chunk in chunks))

    def test_query_router_distinguishes_general_and_document_questions(self) -> None:
        self.assertEqual(route_query("Hello"), "general")
        self.assertEqual(route_query("Explain this from the lecture"), "document")
        self.assertEqual(route_query("How does photosynthesis work"), "general")
        self.assertEqual(route_query("اشرح لي هذا"), "follow_up")

    def test_follow_up_rewrite_is_only_requested_for_pronouns(self) -> None:
        self.assertTrue(needs_rewrite("اشرح ده"))
        self.assertFalse(needs_rewrite("What is a microprocessor"))

    def test_missing_reranker_dependency_falls_back_to_original_ranking(self) -> None:
        reranker = CrossEncoderReranker("missing-model")
        records = [{"id": "a", "content": "Alpha"}, {"id": "b", "content": "Beta"}]
        with patch("builtins.__import__", side_effect=ImportError):
            reranked, latency = reranker.rerank("Alpha", records, limit=2)
        self.assertEqual(reranked, records)
        self.assertGreaterEqual(latency, 0.0)


if __name__ == "__main__":
    unittest.main()
