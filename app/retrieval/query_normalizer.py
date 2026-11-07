"""
app/retrieval/query_normalizer.py

Chuẩn hoá truy vấn của sinh viên TRƯỚC khi embed + rerank (không dùng LLM, không tốn token):

1. Mở rộng viết tắt phổ biến ("bn" -> "bao nhiêu", "đc" -> "được", "ko" -> "không"...).
2. Câu tiếng Việt gõ KHÔNG DẤU -> khôi phục dấu bằng từ điển rút ra từ chính các chunk đã
   index (tần suất từ + cặp từ liền kề, tìm đường tốt nhất kiểu Viterbi). Từ không có biến
   thể có dấu trong từ điển (bfs, dijkstra, tên riêng...) được giữ nguyên.

Chỉ dùng cho truy vấn gửi vào retrieval; câu hỏi gốc vẫn là thứ gửi cho LLM sinh câu trả lời.
Cố ý KHÔNG có bảng đồng nghĩa theo miền ("cây khung" -> "cây bao trùm"): đó là chỉnh theo
bộ test chứ không phải cải thiện hệ thống.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

_VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")
_EN_FUNCTION_WORDS = {
    "the", "is", "are", "was", "what", "how", "why", "which", "who", "when", "where", "does", "do",
    "did", "of", "and", "in", "an", "to", "for", "with", "under", "between", "can", "should", "it",
}

# Viết tắt sinh viên hay dùng, KHÔNG nhập nhằng với từ thật (không có "k", "v", "e", "dc"...).
_ABBREVIATIONS = {
    "bn": "bao nhiêu",
    "bnh": "bao nhiêu",
    "đc": "được",
    "ko": "không",
    "hok": "không",
    "khum": "không",
    "vs": "với",
    "vd": "ví dụ",
    "ntn": "như thế nào",
    "mn": "mọi người",
    "cx": "cũng",
    "nhg": "nhưng",
    "trc": "trước",
    "z": "vậy",
    "j": "gì",
}

_TOKEN = re.compile(r"\w+|\W+", re.UNICODE)
_LEXICON_TTL_SECONDS = 600
_LEXICON_MAX_CHUNKS = 20000


def strip_diacritics(text: str) -> str:
    t = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def has_vietnamese_diacritics(text: str) -> bool:
    return any(ch in _VI_CHARS for ch in (text or "").lower())


def looks_english(text: str) -> bool:
    words = re.findall(r"[a-z]+", (text or "").lower())
    return sum(w in _EN_FUNCTION_WORDS for w in words) >= 2


class Lexicon:
    """Từ điển khôi phục dấu: dạng-không-dấu -> các dạng có thể + tần suất; cặp từ liền kề."""

    def __init__(self) -> None:
        self.variants: dict[str, Counter] = defaultdict(Counter)
        self.bigrams: Counter = Counter()

    def add_text(self, text: str) -> None:
        prev: str | None = None
        for w in re.findall(r"\w+", (text or "").lower()):
            if w.isdigit():
                prev = None
                continue
            self.variants[strip_diacritics(w)][w] += 1
            if prev is not None:
                self.bigrams[(prev, w)] += 1
            prev = w

    def candidates(self, ascii_word: str) -> Counter:
        return self.variants.get(ascii_word, Counter())

    def __len__(self) -> int:
        return len(self.variants)


def build_lexicon(texts) -> Lexicon:
    lex = Lexicon()
    for t in texts:
        lex.add_text(t)
    return lex


def _unigram_score(count: int) -> float:
    return math.log(1 + count)


def restore_diacritics(text: str, lex: Lexicon) -> str:
    """Khôi phục dấu cho các từ ASCII trong `text`. Từ viết HOA (BFS, A*) và từ không có
    trong từ điển được giữ nguyên."""
    tokens = _TOKEN.findall(text)
    word_idx = [i for i, t in enumerate(tokens) if re.fullmatch(r"\w+", t) and not t.isdigit()]
    if not word_idx:
        return text

    # Ứng viên cho từng từ: [(dạng, điểm unigram)]
    cands: list[list[tuple[str, float]]] = []
    for i in word_idx:
        tok = tokens[i]
        low = tok.lower()
        if tok.isupper() and len(tok) > 1:  # viết tắt/ký hiệu: giữ nguyên
            cands.append([(tok, 0.0)])
            continue
        variants = lex.candidates(strip_diacritics(low)) if low.isascii() else Counter()
        if not variants:
            cands.append([(tok, 0.0)])
            continue
        opts = [(form, _unigram_score(cnt)) for form, cnt in variants.items()]
        if low not in variants:  # dạng ASCII nguyên bản chưa từng xuất hiện: vẫn cho phép giữ
            opts.append((low, 0.0))
        cands.append(opts)

    # Viterbi: điểm = unigram + 3 * log(1 + bigram với từ đứng trước)
    best: list[dict[str, tuple[float, str | None]]] = []
    for pos, opts in enumerate(cands):
        layer: dict[str, tuple[float, str | None]] = {}
        for form, uni in opts:
            if pos == 0:
                layer[form] = (uni, None)
                continue
            top_score, top_prev = None, None
            for pform, (pscore, _) in best[pos - 1].items():
                s = pscore + uni + 3.0 * math.log(1 + lex.bigrams.get((pform.lower(), form.lower()), 0))
                if top_score is None or s > top_score:
                    top_score, top_prev = s, pform
            layer[form] = (top_score, top_prev)
        best.append(layer)

    form = max(best[-1], key=lambda f: best[-1][f][0])
    chosen: list[str] = []
    for pos in range(len(cands) - 1, -1, -1):
        chosen.append(form)
        form = best[pos][form][1] or ""
    chosen.reverse()

    for i, new in zip(word_idx, chosen):
        old = tokens[i]
        tokens[i] = _match_case(old, new) if old[:1].isupper() else new
    return "".join(tokens)


def _match_case(old: str, new: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def expand_abbreviations(text: str) -> str:
    out = []
    for tok in _TOKEN.findall(text):
        rep = _ABBREVIATIONS.get(tok.lower()) if re.fullmatch(r"\w+", tok) else None
        out.append(rep if rep and not tok.isupper() else tok)
    return "".join(out)


def normalize_query(query: str, lex: Lexicon | None = None) -> str:
    """Chuẩn hoá truy vấn cho retrieval. Câu tiếng Anh được giữ nguyên."""
    if not query or not query.strip() or looks_english(query):
        return query
    unaccented = not has_vietnamese_diacritics(query)  # xét trên câu GỐC (viết tắt mở ra có dấu)
    out = expand_abbreviations(query)
    if lex is not None and len(lex) and unaccented:
        out = restore_diacritics(out, lex)  # chỉ đụng tới từ ASCII; từ đã có dấu giữ nguyên
    return out


# ---------------------------------------------------------------------------
# Từ điển dùng chung của tiến trình, dựng từ các chunk đã index (làm mới sau TTL)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_cached: tuple[float, Lexicon] | None = None


def get_lexicon() -> Lexicon:
    global _cached
    with _lock:
        if _cached and time.time() - _cached[0] < _LEXICON_TTL_SECONDS:
            return _cached[1]
        try:
            from app.indexing.qdrant_store import iter_chunk_texts

            lex = build_lexicon(iter_chunk_texts(_LEXICON_MAX_CHUNKS))
            logger.info("query_normalizer: từ điển %d từ", len(lex))
        except Exception as exc:  # Qdrant lỗi: bỏ qua bước khôi phục dấu, không chặn retrieval
            logger.warning("query_normalizer: không dựng được từ điển (%s)", exc)
            lex = Lexicon()
        _cached = (time.time(), lex)
        return lex


def reset_lexicon() -> None:
    global _cached
    with _lock:
        _cached = None
