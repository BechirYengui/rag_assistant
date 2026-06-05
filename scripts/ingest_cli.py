"""Batch-ingest local files into the knowledge base.

Usage:
    python -m scripts.ingest_cli sample_docs/*.md path/to/notes.txt report.pdf
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.db import dispose_db, init_db, session_scope
from app.ingestion import ingest_document


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(path.read_bytes()))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


async def main(paths: list[str]) -> None:
    await init_db()
    try:
        for raw in paths:
            path = Path(raw)
            if not path.is_file():
                print(f"skip (not a file): {path}")
                continue
            content = _read(path)
            if not content.strip():
                print(f"skip (empty): {path}")
                continue
            async with session_scope() as session:
                _, count, is_new = await ingest_document(
                    session,
                    content=content,
                    source=str(path),
                    title=path.stem,
                )
            if is_new:
                print(f"ingested {path} -> {count} chunks")
            else:
                print(f"skip (already indexed): {path}")
    finally:
        await dispose_db()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    asyncio.run(main(args))
