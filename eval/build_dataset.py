"""
eval/build_dataset.py

Sinh ỨNG VIÊN câu hỏi từ corpus để bạn duyệt — KHÔNG phải dataset cuối.

    python -m eval.build_dataset --manifest eval/data/corpus_manifest.json \
        --n 60 --out eval/data/questions.candidates.jsonl --seed 42

Quy trình:
 1. Lấy mẫu ngẫu nhiên (có seed) các chunk đủ dài, chia đều theo tài liệu.
 2. LLM sinh câu hỏi + đáp án + bằng chứng trích NGUYÊN VĂN từ chunk.
 3. Kiểm tra bằng CODE rằng bằng chứng thật sự nằm trong chunk (loại câu bịa).
 4. Ghi ra với reviewed=false. Bạn đọc từng câu, sửa, xoá câu dở, đổi reviewed=true.
    run.py mặc định bỏ qua mọi câu reviewed=false.

Hạn chế cần nêu trong báo cáo: câu hỏi sinh từ chunk có xu hướng trùng từ vựng với
chunk => retrieval dễ hơn thực tế. Hãy tự viết thêm >= 15 câu diễn đạt khác và
>= 8 câu answerable=false (hỏi chủ đề tài liệu KHÔNG đề cập).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from app.indexing.qdrant_store import get_all_chunks_for_documents
from app.llm.utils import LLMCallError, acall_structured
from eval.metrics import evidence_match

MIN_CHUNK_CHARS = 350
QUESTION_TYPES = [
    "định nghĩa / khái niệm",
    "so sánh hoặc phân biệt",
    "giải thích tại sao / cơ chế",
    "quy trình / các bước",
    "điều kiện / giới hạn / ngoại lệ",
]

SYSTEM = """Bạn tạo câu hỏi kiểm thử cho hệ thống hỏi đáp tài liệu học tập.
Từ ĐOẠN TÀI LIỆU, viết 1 câu hỏi thuộc loại được yêu cầu mà đoạn đó trả lời được đầy đủ.
Quy tắc:
- Câu hỏi TỰ ĐỦ NGHĨA (không dùng "theo đoạn trên", "ở đây"), như sinh viên hỏi mà chưa thấy tài liệu.
- DIỄN ĐẠT LẠI, tránh sao chép nguyên cụm từ dài trong đoạn.
- gold_answer: 1-3 câu, chỉ dùng thông tin trong đoạn.
- evidence: trích NGUYÊN VĂN 1 câu/cụm (8-40 từ) trong đoạn chứa đáp án — copy chính xác, không sửa."""


class GeneratedQA(BaseModel):
    question: str
    gold_answer: str
    evidence: str = Field(description="Trích nguyên văn từ đoạn tài liệu")


async def main_async(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    pool: list[tuple[str, dict]] = []  # (filename, chunk)
    per_doc: dict[str, list[dict]] = {}
    for d in manifest["documents"]:
        chunks = [c for c in get_all_chunks_for_documents([d["doc_id"]]) if len(c["text"]) >= MIN_CHUNK_CHARS]
        rng.shuffle(chunks)
        per_doc[d["filename"]] = chunks

    # Chia đều theo tài liệu (round-robin)
    while len(pool) < args.n and any(per_doc.values()):
        for fname, chunks in per_doc.items():
            if chunks and len(pool) < args.n * 2:  # lấy dư ~2x vì sẽ có câu bị loại
                pool.append((fname, chunks.pop()))
    if not pool:
        raise SystemExit("Không có chunk đủ dài — kiểm tra corpus/manifest.")

    out_items: list[dict] = []
    for i, (fname, chunk) in enumerate(pool):
        if len(out_items) >= args.n:
            break
        qtype = QUESTION_TYPES[i % len(QUESTION_TYPES)]
        prompt = f"Loại câu hỏi: {qtype}\n\nĐOẠN TÀI LIỆU:\n{chunk['text']}"
        try:
            qa = await acall_structured("quiz", SYSTEM, prompt, GeneratedQA)
        except LLMCallError as exc:
            print(f"[skip] LLM lỗi: {exc}")
            continue
        if not evidence_match(chunk["text"], qa.evidence):
            print(f"[skip] evidence không khớp nguyên văn: {qa.evidence[:60]!r}")
            continue

        out_items.append(
            {
                "id": f"q{len(out_items) + 1:03d}",
                "question": qa.question.strip(),
                "answerable": True,
                "gold_answer": qa.gold_answer.strip(),
                "gold_evidence": [qa.evidence.strip()],
                "source_file": fname,
                "tags": [qtype, "auto-generated"],
                "reviewed": False,
                "review_context": chunk["text"][:800],  # để bạn đối chiếu khi duyệt
            }
        )
        print(f"[ok] {out_items[-1]['id']}: {qa.question[:70]}")

    Path(args.out).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out_items) + "\n", encoding="utf-8"
    )
    print(f"\nĐã ghi {len(out_items)} ứng viên -> {args.out}. HÃY DUYỆT TAY rồi đổi reviewed=true.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval/data/corpus_manifest.json")
    ap.add_argument("--out", default="eval/data/questions.candidates.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()