import struct
from pathlib import Path
from typing import Sequence, List

from .config import (
    CHUNK_LEN, CHUNK_OVERLAP, LINE_BY_LINE_EXT, BLACKLIST_DIRS, BLACKLIST_EXT,
    MAX_FILE_SIZE
    )


def serialize_f32(vec: Sequence[float]) -> bytes:
    """Pack a float list/ndarray into raw little-endian bytes (f32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def chunk_text(text: str, max_len: int = CHUNK_LEN, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of ≈max_len characters.

    Very simple: step = max_len - overlap; slice on character boundaries.
    Guaranteed at least one chunk (even if text < max_len).
    """
    if len(text) <= max_len:
        return [text]

    step = max_len - overlap
    return [text[i : i + max_len] for i in range(0, len(text), step)]


def passages_for_file(path_str: str, content: str) -> List[str]:
    """Return list of passages for *path* based on extension rules."""
    ext = Path(path_str).suffix.lower()
    if ext in LINE_BY_LINE_EXT:
        # One passage per non‑empty line
        return [ln for ln in content.splitlines() if ln.strip()]
    return chunk_text(content)

def is_blacklisted(path: Path) -> bool:
    if path.suffix.lower() in BLACKLIST_EXT:
        return True
    if any(part in BLACKLIST_DIRS for part in path.parts):
        return True
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return True
    except Exception:
        return True
    return False