"""Recursive window text chunker preserving page numbers and source metadata."""

from __future__ import annotations

import re

from ragassistant.schemas.document import DocumentChunk, DocumentPage

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def _split_text_with_separators(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text using hierarchy of separators until slices fit chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    seps = separators or DEFAULT_SEPARATORS
    chosen_sep = ""
    for sep in seps:
        if sep in text:
            chosen_sep = sep
            break

    if not chosen_sep:
        # Fallback to hard character slicing if no separator exists
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    splits = text.split(chosen_sep)
    chunks: list[str] = []
    current_piece = ""

    for piece in splits:
        candidate = f"{current_piece}{chosen_sep}{piece}" if current_piece else piece
        if len(candidate) <= chunk_size:
            current_piece = candidate
        else:
            if current_piece.strip():
                chunks.append(current_piece.strip())
            # If the single piece is larger than chunk_size, recurse on remaining separators
            if len(piece) > chunk_size:
                remaining_seps = [s for s in seps if s != chosen_sep]
                sub_chunks = _split_text_with_separators(piece, chunk_size, chunk_overlap, remaining_seps)
                chunks.extend(sub_chunks)
                current_piece = ""
            else:
                current_piece = piece

    if current_piece.strip():
        chunks.append(current_piece.strip())

    # Build overlapping windows between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        merged_with_overlap: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_prefix = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
            merged_with_overlap.append(f"{overlap_prefix} ... {chunks[i]}")
        return merged_with_overlap

    return chunks


def chunk_document_pages(
    pages: list[DocumentPage],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """Slice document pages into structured DocumentChunk models with traceable chunk IDs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

    all_chunks: list[DocumentChunk] = []
    global_index = 0

    for page in pages:
        raw_splits = _split_text_with_separators(page.text, chunk_size, chunk_overlap)
        doc_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", page.doc_name)

        for chunk_idx, text_content in enumerate(raw_splits):
            clean_text = text_content.strip()
            if not clean_text:
                continue

            chunk_id = f"{doc_stem}_p{page.page_number}_c{chunk_idx}"
            # Approximate token count (~4 chars per token)
            tokens = max(1, len(clean_text) // 4)

            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    text=clean_text,
                    token_count=tokens,
                )
            )
            global_index += 1

    return all_chunks
