# Omar’s Library

A local assistant for asking questions about your documents in Arabic or English. Upload a file, choose what to search, and open the cited PDF page to check the answer. Your documents, chat history, and model inference stay on your machine with the default local setup.

## What it does

- Reads PDF, DOCX, CSV, text, Markdown, and supported source files.
- Combines keyword search (BM25) with semantic search (bge-m3 and ChromaDB).
- Shows cited excerpts and links to PDF pages.

## Run locally on Windows

Install Python 3.11+, Node.js 22.12+, and Ollama. With Ollama running, open PowerShell in the project folder:

```powershell
ollama pull qwen2.5:latest
ollama pull bge-m3
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, from the project folder:

```powershell
cd frontend
npm ci
npm run dev
```

Open [the local app](http://127.0.0.1:5173/). Import a file from **Your library**, select it next to the message box, and ask a question. You can try [this sample document](examples/atlas.md).

## Project notes

- React and TypeScript power the interface. FastAPI serves the API, and SQLite stores chat history.
- See [.env.example](.env.example) for settings. To load a copied `.env` file, add `--env-file .env` to the Uvicorn command.
- See the [evaluation notes](evaluation/RESULTS.md) for retrieval and answer-quality results.

## Checks

Run `python -m unittest discover -s tests` from the project folder with the
virtual environment active. From `frontend`, run `npm run lint` and
`npm run build`.
