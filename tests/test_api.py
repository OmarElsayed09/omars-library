import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import api


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir()
        self.chat_db = self.root / "chats.sqlite3"
        self.chat_db_patch = patch.object(api, "CHAT_DB_PATH", self.chat_db)
        self.docs_dir_patch = patch.object(api.agent, "DOCS_DIR", self.docs_dir)
        self.chat_db_patch.start()
        self.docs_dir_patch.start()
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        self.client.close()
        self.docs_dir_patch.stop()
        self.chat_db_patch.stop()
        self.temp_dir.cleanup()

    def create_chat(self) -> str:
        response = self.client.post("/api/chats")
        self.assertEqual(response.status_code, 200)
        return response.json()["id"]

    def test_chat_turn_is_persisted_with_answer(self) -> None:
        chat_id = self.create_chat()

        with (
            patch.object(api.agent.collection, "count", return_value=0),
            patch.object(api.agent, "ask", return_value="Hello there.") as ask,
        ):
            response = self.client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Explain gravity", "language": "en"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Hello there.")
        self.assertEqual(ask.call_args.args[0][-1], {"role": "user", "content": "Explain gravity"})
        self.assertNotIn("call search_docs", ask.call_args.args[0][0]["content"])

        messages = self.client.get(f"/api/chats/{chat_id}/messages").json()
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], "Hello there.")
        self.assertEqual(messages[1]["sources"], [])

        chats = self.client.get("/api/chats").json()
        self.assertEqual(chats[0]["title"], "Explain gravity")

    def test_blank_chat_message_is_rejected(self) -> None:
        chat_id = self.create_chat()

        response = self.client.post(
            f"/api/chats/{chat_id}/messages",
            json={"message": "   ", "language": "auto"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get(f"/api/chats/{chat_id}/messages").json(), [])

    def test_chat_message_requires_supported_language(self) -> None:
        chat_id = self.create_chat()

        response = self.client.post(
            f"/api/chats/{chat_id}/messages",
            json={"message": "Hello", "language": "fr"},
        )

        self.assertEqual(response.status_code, 422)

    def test_arabic_reply_language_overrides_default_english_instruction(self) -> None:
        chat_id = self.create_chat()

        with patch.object(api.agent.collection, "count", return_value=0):
            messages, _ = api.prepare_chat_turn(chat_id, "اشرح هذا", "ar")

        system_prompt = messages[0]["content"]
        self.assertIn("أجب بالعربية", system_prompt)

    def test_auto_reply_language_prioritizes_latest_message_language(self) -> None:
        chat_id = self.create_chat()

        with patch.object(api.agent.collection, "count", return_value=0):
            messages, _ = api.prepare_chat_turn(chat_id, "اشرح هذا", "auto")

        system_prompt = messages[0]["content"]
        self.assertIn("أجب بالعربية", system_prompt)

    def test_stream_repairs_wrong_script_before_sending_or_saving(self) -> None:
        chat_id = self.create_chat()
        bad = SimpleNamespace(message=SimpleNamespace(content="中文"))
        good = SimpleNamespace(message=SimpleNamespace(content="أهلاً بيك!"))
        with patch.object(api.agent.ollama_client, "chat", side_effect=[iter([bad]), good]):
            reply = self.client.post(f"/api/chats/{chat_id}/messages/stream", json={"message": "أهلاً"})
        self.assertNotIn("中文", reply.text)
        self.assertIn("أهلاً بيك!", reply.text)
        saved = self.client.get(f"/api/chats/{chat_id}/messages").json()
        self.assertEqual(saved[-1]["content"], "أهلاً بيك!")

    @unittest.skipUnless(os.getenv("LIVE_TERM_DEMO") == "1", "requires local Ollama")
    def test_live_general_term_uses_context_and_skips_unsolicited_code(self) -> None:
        for question, terms in [
            ("rag", ("retrieval", "generation")),
            ("In a project status report, what do the RAG colors stand for?", ("red", "amber", "green")),
        ]:
            with self.subTest(question=question):
                chat = self.create_chat()
                reply = self.client.post(f"/api/chats/{chat}/messages/stream", json={"message": question})
                events = [json.loads(line[6:]) for line in reply.text.splitlines() if line.startswith("data: ")]
                self.assertEqual(events[-1]["type"], "done")
                answer = "".join(e["content"] for e in events if e["type"] == "delta").lower()
                for term in terms:
                    self.assertIn(term, answer)
                self.assertNotIn("```", answer)

    def test_uploaded_document_filename_is_reduced_to_basename(self) -> None:
        with (
            patch("api.index_with_http_errors") as index_documents,
            patch(
                "api.list_documents",
                return_value=api.LibraryStatus(documents=[], chunks=0),
            ),
        ):
            response = self.client.post(
                "/api/documents/upload",
                files=[("files", ("../../notes.txt", b"document text", "text/plain"))],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual((self.docs_dir / "notes.txt").read_bytes(), b"document text")
        index_documents.assert_called_once_with()

    def test_upload_rejects_unsupported_document_type(self) -> None:
        with patch("api.index_with_http_errors") as index_documents:
            response = self.client.post(
                "/api/documents/upload",
                files=[("files", ("script.exe", b"not a document", "application/octet-stream"))],
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(list(self.docs_dir.iterdir()), [])
        index_documents.assert_not_called()

    def test_document_lookup_rejects_paths_outside_document_directory(self) -> None:
        outside_document = self.root / "outside.txt"
        outside_document.write_text("not in docs", encoding="utf-8")

        with self.assertRaises(HTTPException) as raised:
            api.get_document_path("..\\outside.txt")

        self.assertEqual(raised.exception.status_code, 404)

    def test_pdf_preview_returns_inline_original_only_for_library_pdfs(self) -> None:
        name = "جلسة 1.pdf"
        original = b"%PDF-1.7\n"
        (self.docs_dir / name).write_bytes(original)
        response = self.client.get("/api/documents/pdf", params={"source": name})
        self.assertEqual(response.content, original)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertTrue(response.headers["content-disposition"].startswith("inline"))
        (self.docs_dir / "notes.txt").write_text("notes")
        self.assertEqual(self.client.get("/api/documents/pdf", params={"source": "notes.txt"}).status_code, 415)
        self.assertEqual(self.client.get("/api/documents/pdf", params={"source": "gone.pdf"}).status_code, 404)

    def test_streamed_chat_sends_tokens_and_persists_completed_turn(self) -> None:
        chat_id = self.create_chat()
        chunks = [
            SimpleNamespace(message=SimpleNamespace(content="Hello")),
            SimpleNamespace(message=SimpleNamespace(content=" there.")),
        ]

        with (
            patch.object(api.agent.collection, "count", return_value=0),
            patch.object(api.agent.ollama_client, "chat", return_value=iter(chunks)) as chat,
        ):
            response = self.client.post(
                f"/api/chats/{chat_id}/messages/stream",
                json={"message": "Explain gravity", "language": "en"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"content": "Hello"', response.text)
        self.assertIn('"content": " there."', response.text)
        completion = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")][-1]
        self.assertEqual(completion["type"], "done")
        self.assertTrue(chat.call_args.kwargs["stream"])
        messages = self.client.get(f"/api/chats/{chat_id}/messages").json()
        self.assertEqual([message["content"] for message in messages], ["Explain gravity", "Hello there."])
        self.assertEqual(completion["message_id"], messages[-1]["id"])

    def test_health_endpoint_reports_local_services(self) -> None:
        with patch.object(api.agent.ollama_client, "list", return_value=SimpleNamespace(models=[])):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("status", payload)
        self.assertIn("chunk_count", payload)
        self.assertIn("reranker_enabled", payload)

    def test_answer_feedback_is_private_to_an_assistant_message_and_can_be_replaced(self) -> None:
        chat_id = self.create_chat()
        with api.database() as connection:
            connection.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, 'assistant', 'answer')",
                (chat_id,),
            )
            message_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]

        first = self.client.put(
            f"/api/chats/{chat_id}/messages/{message_id}/feedback",
            json={"rating": "up", "comment": "Useful source"},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["helpful_rate"], 1.0)

        second = self.client.put(
            f"/api/chats/{chat_id}/messages/{message_id}/feedback",
            json={"rating": "down"},
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json(), {
            "total": 1, "helpful": 0, "not_helpful": 1, "helpful_rate": 0.0,
        })
        self.assertEqual(self.client.get("/api/feedback").json()["total"], 1)

    def test_feedback_rejects_user_messages(self) -> None:
        chat_id = self.create_chat()
        with api.database() as connection:
            connection.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', 'question')",
                (chat_id,),
            )
            message_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
        response = self.client.put(
            f"/api/chats/{chat_id}/messages/{message_id}/feedback", json={"rating": "up"},
        )
        self.assertEqual(response.status_code, 404)

    def test_backup_exports_chat_history_and_documents(self) -> None:
        chat_id = self.create_chat()
        with api.database() as connection:
            connection.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', 'question')",
                (chat_id,),
            )
        (self.docs_dir / "notes.txt").write_text("private notes", encoding="utf-8")

        response = self.client.get("/api/backup")

        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            backup = json.loads(archive.read("backup.json"))
            self.assertEqual(backup["version"], 1)
            self.assertEqual(backup["chats"][0]["messages"][0]["content"], "question")
            self.assertEqual(archive.read("docs/notes.txt"), b"private notes")

    def test_restore_merges_without_overwriting_existing_data(self) -> None:
        with api.database() as connection:
            connection.execute(
                """
                INSERT INTO chats (id, title) VALUES ('existing', 'Keep this conversation')
                """
            )
        existing_doc = self.docs_dir / "existing.txt"
        existing_doc.write_text("keep this file", encoding="utf-8")
        backup_data = {
            "version": 1,
            "chats": [
                {
                    "id": "existing",
                    "title": "Overwritten title",
                    "created_at": "2026-01-01 00:00:00",
                    "updated_at": "2026-01-01 00:00:00",
                    "messages": [],
                },
                {
                    "id": "imported",
                    "title": "Imported conversation",
                    "created_at": "2026-01-01 00:00:00",
                    "updated_at": "2026-01-01 00:00:00",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Imported answer",
                            "sources": [],
                            "created_at": "2026-01-01 00:00:00",
                        }
                    ],
                },
            ],
        }
        archive_data = io.BytesIO()
        with zipfile.ZipFile(archive_data, "w") as archive:
            archive.writestr("backup.json", json.dumps(backup_data))
            archive.writestr("docs/existing.txt", "do not replace")
            archive.writestr("docs/new.txt", "new document")

        with patch.object(api.agent, "index_docs"):
            response = self.client.post(
                "/api/backup/restore",
                files=[("file", ("backup.zip", archive_data.getvalue(), "application/zip"))],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"documents_added": 1, "chats_added": 1})
        self.assertEqual(existing_doc.read_text(encoding="utf-8"), "keep this file")
        self.assertEqual((self.docs_dir / "new.txt").read_text(encoding="utf-8"), "new document")
        chats = {chat["id"]: chat["title"] for chat in self.client.get("/api/chats").json()}
        self.assertEqual(chats["existing"], "Keep this conversation")
        self.assertEqual(chats["imported"], "Imported conversation")

    def test_restore_rejects_archive_paths_outside_docs(self) -> None:
        archive_data = io.BytesIO()
        with zipfile.ZipFile(archive_data, "w") as archive:
            archive.writestr("backup.json", json.dumps({"version": 1, "chats": []}))
            archive.writestr("docs/../outside.txt", "unsafe")

        response = self.client.post(
            "/api/backup/restore",
            files=[("file", ("backup.zip", archive_data.getvalue(), "application/zip"))],
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse((self.root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
