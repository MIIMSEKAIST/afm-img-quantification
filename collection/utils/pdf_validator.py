from pathlib import Path


def is_valid_pdf(path: Path, min_bytes: int = 5000) -> tuple:
    """PDF 최소 유효성 검사. Returns: (is_valid, reason)"""
    if not path.exists():
        return False, "file_not_found"

    size = path.stat().st_size
    if size < min_bytes:
        return False, f"too_small ({size} bytes)"

    with open(path, "rb") as f:
        header = f.read(8)
        if not header.startswith(b"%PDF"):
            return False, "not_pdf"

        f.seek(max(0, size - 128))
        tail = f.read()
        if b"%%EOF" not in tail:
            return False, "truncated"

    return True, "ok"