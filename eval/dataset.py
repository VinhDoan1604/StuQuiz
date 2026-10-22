"""
eval/dataset.py

Định dạng JSONL, mỗi dòng 1 câu hỏi:

{"id": "q001",
 "question": "...",
 "answerable": true,                       # false = tài liệu KHÔNG chứa câu trả lời
 "gold_answer": "...",                     # bắt buộc nếu answerable
 "gold_evidence": ["trích NGUYÊN VĂN ..."],# bắt buộc nếu answerable; rỗng nếu không
 "source_file": "slide-bai-3.pdf",         # tuỳ chọn, để đọc cho dễ
 "tags": ["definition"],                   # tuỳ chọn, để phân tích theo nhóm
 "reviewed": true}                         # build_dataset.py ghi false; bạn duyệt xong mới đổi true

Câu hỏi không trả lời được (answerable=false) rất quan trọng: chúng đo đúng điều
README tuyên bố — hệ thống biết nói "tài liệu không đề cập" thay vì bịa.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    answerable: bool
    gold_answer: str | None = None
    gold_evidence: tuple[str, ...] = ()
    source_file: str | None = None
    tags: tuple[str, ...] = ()
    reviewed: bool = True


def load_dataset(path: str | Path, include_unreviewed: bool = False) -> list[EvalItem]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Không thấy dataset: {path}")

    items: list[EvalItem] = []
    seen: set[str] = set()
    skipped_unreviewed = 0

    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path.name}:{ln}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: JSON không hợp lệ ({exc})") from exc

        qid = str(obj.get("id", "")).strip()
        question = str(obj.get("question", "")).strip()
        if not qid or not question:
            raise DatasetError(f"{where}: thiếu 'id' hoặc 'question'.")
        if qid in seen:
            raise DatasetError(f"{where}: id '{qid}' bị trùng.")
        seen.add(qid)

        answerable = obj.get("answerable", True)
        if not isinstance(answerable, bool):
            raise DatasetError(f"{where}: 'answerable' phải là true/false.")
        evidence = tuple(str(e).strip() for e in obj.get("gold_evidence", []) if str(e).strip())
        gold_answer = (obj.get("gold_answer") or "").strip() or None

        if answerable and not (gold_answer and evidence):
            raise DatasetError(f"{where} ({qid}): câu trả lời được cần 'gold_answer' và 'gold_evidence'.")
        if not answerable and evidence:
            raise DatasetError(f"{where} ({qid}): câu không trả lời được không được có 'gold_evidence'.")

        reviewed = bool(obj.get("reviewed", True))
        if not reviewed and not include_unreviewed:
            skipped_unreviewed += 1
            continue

        items.append(
            EvalItem(
                id=qid,
                question=question,
                answerable=answerable,
                gold_answer=gold_answer,
                gold_evidence=evidence,
                source_file=obj.get("source_file"),
                tags=tuple(obj.get("tags", [])),
                reviewed=reviewed,
            )
        )

    if skipped_unreviewed:
        print(f"[dataset] Bỏ qua {skipped_unreviewed} câu chưa duyệt (reviewed=false).")
    if not items:
        raise DatasetError("Dataset rỗng sau khi lọc.")
    return items