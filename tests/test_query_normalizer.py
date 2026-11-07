from app.retrieval.query_normalizer import (
    build_lexicon,
    expand_abbreviations,
    normalize_query,
    strip_diacritics,
)

CORPUS = [
    "Đồ thị hai phía có thể kiểm tra bằng thuật toán BFS. Đồ thị có hướng không chu trình.",
    "Kiểm tra đồ thị bằng duyệt theo chiều rộng. Làm sao dùng thuật toán này để kiểm tra.",
    "Để kiểm tra tính hai phía của đồ thị, tô màu hai phía.",
    "The binary search tree supports search in O(log n).",
]


def lex():
    return build_lexicon(CORPUS)


def test_strip_diacritics():
    assert strip_diacritics("Đồ thị hai phía") == "Do thi hai phia"


def test_abbreviations_expand_but_acronyms_and_english_stay():
    assert expand_abbreviations("kruskal bn vậy mn") == "kruskal bao nhiêu vậy mọi người"
    assert expand_abbreviations("BN là gì") == "BN là gì"  # viết hoa = ký hiệu
    assert normalize_query("How does it work for the tree?", lex()) == "How does it work for the tree?"


def test_restores_diacritics_using_corpus_lexicon():
    out = normalize_query("lam sao dung bfs de kiem tra do thi hai phia", lex())
    assert out == "làm sao dùng bfs để kiểm tra đồ thị hai phía"


def test_unknown_words_and_acronyms_kept():
    out = normalize_query("dijkstra va BFS tren do thi", lex())
    assert "dijkstra" in out and "BFS" in out and "đồ thị" in out


def test_accented_query_not_touched_except_abbreviations():
    assert normalize_query("đồ thị hai phía là gì ko", lex()) == "đồ thị hai phía là gì không"


def test_empty_lexicon_still_expands_abbreviations():
    assert normalize_query("lam sao ko", build_lexicon([])) == "lam sao không"
