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
    """Recursively split text using hierarchy of separators into verbatim chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    seps = separators or DEFAULT_SEPARATORS
    chosen_sep = ""
    for sep in seps:
        if sep in text:
            chosen_sep = sep
            break

    if not chosen_sep:
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    splits = text.split(chosen_sep)
    chunks: list[str] = []
    current_pieces: list[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece) + (len(chosen_sep) if current_pieces else 0)
        if current_length + piece_len <= chunk_size:
            current_pieces.append(piece)
            current_length += piece_len
        else:
            if current_pieces:
                combined = chosen_sep.join(current_pieces).strip()
                if combined:
                    chunks.append(combined)

                overlap_pieces: list[str] = []
                overlap_len = 0
                for p in reversed(current_pieces):
                    add_len = len(p) + (len(chosen_sep) if overlap_pieces else 0)
                    if overlap_len + add_len <= chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += add_len
                    else:
                        break
                current_pieces = overlap_pieces
                current_length = overlap_len

            if len(piece) > chunk_size:
                remaining_seps = [s for s in seps if s != chosen_sep]
                sub_chunks = _split_text_with_separators(piece, chunk_size, chunk_overlap, remaining_seps)
                chunks.extend(sub_chunks)
                current_pieces = []
                current_length = 0
            else:
                add_len = len(piece) + (len(chosen_sep) if current_pieces else 0)
                current_pieces.append(piece)
                current_length += add_len

    if current_pieces:
        combined = chosen_sep.join(current_pieces).strip()
        if combined:
            chunks.append(combined)

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
