import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import agent
from rag.generation.quality import casual_reply, quality_issue
from rag.ingestion import loaders


class ChunkTests(unittest.TestCase):
    def test_quality_guard_preserves_inline_code_and_rejects_citation_only(self):
        self.assertTrue(quality_issue("[1]", "ar", 1, "اشرح"))
        self.assertFalse(quality_issue("استخدم `items[1]`.", "ar", 0, "اشرح"))

    def test_grounded_answer_requires_a_valid_citation(self) -> None:
        self.assertIn("cite", quality_issue("The answer is 42.", "en", 2, "What is it?").lower())
        self.assertEqual(quality_issue("The answer is 42 [1].", "en", 2, "What is it?"), "")

    def test_unexpected_cyrillic_triggers_a_repair(self) -> None:
        self.assertIn("Cyrillic", quality_issue("الإجابة обратي [1]", "ar", 1, "اشرح"))

    def test_short_arabic_greeting_is_local_and_natural(self) -> None:
        self.assertEqual(casual_reply("ازيك؟", "ar"), "أهلًا! أنا كويس، شكرًا لسؤالك. إزاي أقدر أساعدك؟")

    def test_ocr_replaces_nearly_empty_pdf_page_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scan.pdf"
            path.write_bytes(b"placeholder")
            page = type("Page", (), {"extract_text": lambda self: ""})()
            reader = type("Reader", (), {"pages": [page]})()
            with (
                patch("rag.ingestion.loaders.PdfReader", return_value=reader),
                patch("rag.ingestion.loaders.OCR_ENABLED", True),
                patch("rag.ingestion.loaders.ocr_pdf_page", return_value="نص ممسوح") as ocr,
            ):
                self.assertEqual(loaders.read_pdf(path), [(1, "نص ممسوح")])
                ocr.assert_called_once_with(path, 0)

    def test_multi_query_retrieval_keeps_evidence_from_each_query(self) -> None:
        first = [{"source": "lecture.pdf", "chunk": 1, "content": "first"}]
        second = [{"source": "lecture.pdf", "chunk": 2, "content": "second"}]
        trace = {"dense": [], "bm25": [], "fused": [], "reranked": False, "selected": []}
        with patch.object(agent, "retrieve_docs_with_trace", side_effect=[(first, trace), (second, trace)]):
            records, multi_trace = agent.retrieve_docs_for_queries(["original", "translated"], limit=2)
        self.assertEqual([record["chunk"] for record in records], [1, 2])
        self.assertEqual(multi_trace["queries"], ["original", "translated"])

    def test_chunks_respect_size_and_overlap(self) -> None:
        self.assertEqual(
            agent.chunk("abcdefghij", size=4, overlap=1),
            ["abcd", "defg", "ghij"],
        )

    def test_empty_or_whitespace_text_produces_no_chunks(self) -> None:
        self.assertEqual(agent.chunk(""), [])
        self.assertEqual(agent.chunk(" \n\t "), [])

    def test_chunks_do_not_contain_only_whitespace(self) -> None:
        chunks = agent.chunk("alpha   beta\n gamma", size=8, overlap=2)

        self.assertTrue(chunks)
        self.assertTrue(all(part.strip() for part in chunks))

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        for size, overlap in ((0, 0), (4, -1), (4, 4), (4, 5)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    agent.chunk("some text", size=size, overlap=overlap)

    def test_retrieval_reranks_semantically_close_results_by_term_overlap(self) -> None:
        result = {
            "documents": [[
                "A general guide to kitchen appliances.",
                "Reset the router password from network settings.",
                "A tangential account of unrelated subjects.",
            ]],
            "metadatas": [[
                {"source": "kitchen.txt", "chunk": 1},
                {"source": "network.txt", "chunk": 2},
                {"source": "unrelated.txt", "chunk": 3},
            ]],
            "distances": [[0.1, 0.2, 1.5]],
        }
        with (
            patch.object(agent.collection, "count", return_value=3),
            patch.object(agent, "embed", return_value=[[0.0]]),
            patch.object(agent.collection, "query", return_value=result),
        ):
            records = agent.retrieve_docs("router password")

        self.assertEqual(records[0]["source"], "network.txt")
        self.assertEqual(records[0]["chunk"], 2)
        self.assertNotIn("unrelated.txt", [record["source"] for record in records])

    def test_irrelevant_results_are_removed_when_semantic_and_lexical_scores_are_low(self) -> None:
        result = {
            "documents": [["A discussion of a distant unrelated topic."]],
            "metadatas": [[{"source": "unrelated.txt", "chunk": 1}]],
            "distances": [[1.31]],
        }
        with (
            patch.object(agent.collection, "count", return_value=1),
            patch.object(agent, "embed", return_value=[[0.0]]),
            patch.object(agent.collection, "query", return_value=result),
        ):
            records = agent.retrieve_docs("quantum agriculture")

        self.assertEqual(records, [])

    def test_arabic_search_terms_normalize_diacritics_and_alef_variants(self) -> None:
        self.assertEqual(
            agent.search_terms("إِدارةُ الْمَشْرُوع"),
            agent.search_terms("اداره المشروع"),
        )

    def test_search_terms_normalize_common_english_plurals(self) -> None:
        self.assertEqual(agent.search_terms("addresses"), {"address"})
        self.assertEqual(agent.search_terms("activities"), {"activity"})
        self.assertEqual(agent.search_terms("bytes"), {"byte"})

    def test_arabic_term_overlap_keeps_relevant_result(self) -> None:
        result = {
            "documents": [["إدارة المشروع تتطلب تخطيطًا ومتابعة مستمرة."]],
            "metadatas": [[{"source": "project.txt", "chunk": 1}]],
            "distances": [[1.4]],
        }
        with (
            patch.object(agent.collection, "count", return_value=1),
            patch.object(agent, "embed", return_value=[[0.0]]),
            patch.object(agent.collection, "query", return_value=result),
        ):
            records = agent.retrieve_docs("ادارة المشروع")

        self.assertEqual([record["source"] for record in records], ["project.txt"])


if __name__ == "__main__":
    unittest.main()
