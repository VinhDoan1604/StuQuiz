# STATUS: READY
"""
Fixed-size chunking (baseline) — dùng để so sánh benchmark với semantic
chunker (app/chunking/semantic_chunker.py, STUB). Không phải chiến lược
chính thức của project nhưng cần có để đo lường "semantic chunking tốt
hơn bao nhiêu %" trong README.
"""
import uuid
from app.models.schemas import Chunk, ChunkMetadata

try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


def fixed_size_chunk(
    text: str,
    source_file: str,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    if ENCODER is not None:
        tokens = ENCODER.encode(text)
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        doc_id = str(uuid.uuid4())
        while start < len(tokens):
            end = min(start + chunk_size_tokens, len(tokens))
            piece = ENCODER.decode(tokens[start:end])
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    text=piece,
                    token_count=end - start,
                    chunk_index=chunk_index,
                    metadata=ChunkMetadata(
                        source_file=source_file,
                        content_type="content",
                    ),
                )
            )
            chunk_index += 1
            if end == len(tokens):
                break
            start = end - overlap_tokens
        return chunks
    else:
        words = text.split()
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        doc_id = str(uuid.uuid4())
        chunk_size_words = max(1, int(chunk_size_tokens / 1.3))
        overlap_words = max(1, int(overlap_tokens / 1.3))
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            piece = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    text=piece,
                    token_count=int((end - start) * 1.3),
                    chunk_index=chunk_index,
                    metadata=ChunkMetadata(
                        source_file=source_file,
                        content_type="content",
                    ),
                )
            )
            chunk_index += 1
            if end == len(words):
                break
            start = end - overlap_words
        return chunks
