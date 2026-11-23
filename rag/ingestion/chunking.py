def recursive_text_split(
    text: str,
    size: int = 800,
    overlap: int = 120,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "؟", "؛", " "),
) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")

    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # Prefer natural boundaries in the second half, always making progress.
            lower = start + max(size // 2, overlap)
            for separator in separators:
                pos = text.rfind(separator, lower, end)
                if separator and pos >= lower:
                    end = pos + len(separator)
                    break
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        start = end - overlap
    return chunks
