"""Run bilingual questions through the same streaming endpoint as the browser."""
import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "evaluation"


def turn(client, chat, source, question):
    start = time.perf_counter()
    response = client.post(f"/api/chats/{chat}/messages/stream",
                           json={"message": question, "source": source})
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    if events[-1]["type"] != "done":
        raise RuntimeError(events[-1])
    pages = [s.get("page") for s in events[0]["sources"] if s["source"] == source]
    answer = "".join(e["content"] for e in events if e["type"] == "delta")
    return answer, pages, round(time.perf_counter() - start, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not all((EVAL / "lectures" / f"MIT18_06S10_L{n:02}.pdf").exists() for n in (1, 2, 3)):
        parser.error("Run evaluation/fetch_lectures.ps1 first.")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ.update(DOCS_DIR=str(EVAL / "lectures"), DB_DIR=str(Path(tmp) / "vectors"),
                          CHAT_DB_PATH=str(Path(tmp) / "chats.sqlite3"))
        import api
        api.agent.index_docs()
        rows = []
        with TestClient(api.app) as client:
            for split, number, question, gold, pages, *context in json.loads(
                    (EVAL / "lecture_cases.json").read_text(encoding="utf-8")):
                if args.split not in ("all", split):
                    continue
                source = f"MIT18_06S10_L{number:02}.pdf"
                chat = client.post("/api/chats").json()["id"]
                for prior in context:
                    turn(client, chat, source, prior)
                answer, retrieved, seconds = turn(client, chat, source, question)
                rows.append(dict(split=split, source=source, question=question, expected=gold,
                                 pages=pages, retrieved=retrieved, answer=answer, seconds=seconds,
                                 page_hit=bool(set(pages) & set(retrieved)) if pages else None))
                print(f"{len(rows)} completed ({seconds}s)", flush=True)
        hits = [r["page_hit"] for r in rows if r["page_hit"] is not None]
        report = dict(model=api.agent.LLM, embeddings=api.agent.EMBED, split=args.split,
                      page_hits=sum(hits), answerable=len(hits),
                      median_seconds=statistics.median(r["seconds"] for r in rows), cases=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
