import csv
import io
import os
from pathlib import Path

from pypdf import PdfReader

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".java", ".go", ".rs", ".cpp", ".cs"}
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
OCR_LANG = os.getenv("OCR_LANG", "ara+eng")
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "40"))
OCR_TESSDATA_DIR = os.getenv("OCR_TESSDATA_DIR", "")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")


def ocr_pdf_page(path: Path, page_index: int) -> str:
    """Extract a scanned PDF page locally, only when OCR is explicitly enabled."""
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as error:
        raise RuntimeError(
            "OCR requires pypdfium2 and pytesseract. Install the optional OCR dependencies."
        ) from error
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    config = ""
    if OCR_TESSDATA_DIR:
        config = f'--tessdata-dir "{Path(OCR_TESSDATA_DIR).resolve()}"'
    try:
        document = pdfium.PdfDocument(str(path))
        page = document[page_index]
        image = page.render(scale=2).to_pil()
        return pytesseract.image_to_string(image, lang=OCR_LANG, config=config).strip()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("OCR is enabled but Tesseract is not installed or not on PATH.") from error


def read_txt(path: Path) -> list[tuple[int | None, str]]:
    return [(None, path.read_text(encoding="utf-8"))]


def read_pdf(path: Path) -> list[tuple[int | None, str]]:
    pages = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if OCR_ENABLED and len(text) < OCR_MIN_TEXT_CHARS:
            text = ocr_pdf_page(path, page_number - 1) or text
        pages.append((page_number, text))
    return pages


def read_markdown(path: Path) -> list[tuple[int | None, str]]:
    return [(None, path.read_text(encoding="utf-8"))]


def read_csv(path: Path) -> list[tuple[int | None, str]]:
    output = io.StringIO()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        for row_number, row in enumerate(reader, start=1):
            output.write(f"Row {row_number}: " + " | ".join(row) + "\n")
    return [(None, output.getvalue())]


def read_docx(path: Path) -> list[tuple[int | None, str]]:
    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError("DOCX support requires the python-docx package.") from error
    document = Document(path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            paragraphs.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [(None, "\n".join(paragraphs))]


def read_document(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        return [(None, "\n".join(f"{i}: {line}" for i, line in enumerate(lines, 1)))]
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".csv":
        return read_csv(path)
    if suffix in {".md", ".markdown"}:
        return read_markdown(path)
    if suffix == ".txt":
        return read_txt(path)
    raise ValueError(f"Unsupported document type: {suffix}")
