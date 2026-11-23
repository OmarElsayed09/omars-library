import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
from fastapi.testclient import TestClient

import api
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.hybrid import HybridRetriever


class DocumentScopeTests(unittest.TestCase):
    def exercise_scope(self, live=False):
        # A separate collection and chat database keep the user's library out of the demo.
        import uuid
        collection = chromadb.EphemeralClient().create_collection("scope-" + uuid.uuid4().hex)
        index = BM25Index()
        examples = Path(__file__).resolve().parents[1] / "examples"
        with tempfile.TemporaryDirectory() as temp, (
            patch.object(api, "CHAT_DB_PATH", Path(temp) / "chat.sqlite3")
        ), patch.object(api.agent, "DOCS_DIR", examples), patch.object(
            api.agent, "collection", collection
        ), patch.object(api.agent, "bm25_index", index), patch.object(
            api.agent, "hybrid_retriever", HybridRetriever(index)
        ), TestClient(api.app) as client:
            api.agent.index_docs()
            chat_id = client.post("/api/chats").json()["id"]
            for source, expected, other, endpoint in [
                ("atlas.md", "A-204", "C-318", "messages"),
                ("cedar.md", "C-318", "A-204", "messages/stream"),
            ]:
                res = client.post(f"/api/chats/{chat_id}/{endpoint}", json={
                    "message": "الورشة في قاعة إيه؟ اكتب كود القاعة زي ما هو.",
                    "source": source,
                })
                self.assertEqual(res.status_code, 200, res.text)
                if endpoint.endswith("stream"):
                    events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]
                    self.assertEqual(events[-1]["type"], "done")
                    sources = events[0]["sources"]
                    answer = "".join(e["content"] for e in events if e["type"] == "delta")
                else:
                    sources = res.json()["sources"]
                    answer = res.json()["answer"]
                self.assertEqual({s["source"] for s in sources}, {source})
                self.assertIn(expected, sources[0]["content"])
                if live:
                    self.assertIn(expected, answer, answer)
                    self.assertNotIn(other, answer, answer)
            self.assertEqual(len(client.get(f"/api/chats/{chat_id}/messages").json()), 4)

    def test_source_isolation_on_both_endpoints(self):
        def model_reply(*, messages, model, stream=False, options=None):
            if options and "num_predict" in options:
                return SimpleNamespace(message=SimpleNamespace(content="What is the workshop room code?"))
            self.assertNotIn("tools", messages[0])
            self.assertEqual(len(messages), 2)
            self.assertNotIn("A-204" if stream else "C-318", messages[0]["content"])
            chunk = SimpleNamespace(message=SimpleNamespace(content="القاعة [1]."))
            return iter([chunk]) if stream else chunk

        with patch.object(api.agent, "embed", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]), patch.object(
            api.agent.ollama_client, "chat", side_effect=model_reply
        ):
            self.exercise_scope()

    @unittest.skipUnless(os.getenv("LIVE_SCOPE_DEMO") == "1", "requires local Ollama models")
    def test_live_arabic_question_on_two_english_sources(self):
        self.exercise_scope(live=True)
