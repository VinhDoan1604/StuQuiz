"""Test logic mở rộng parent-chunk (thuần, không cần DB). pytest tests/test_expansion.py -v"""
from app.retrieval.expansion import expand_chunks_with_parents, make_snippet


def ct(s: str) -> int:  # đếm token đơn giản: số từ
    return len(s.split())


def child(i, pid, text="child text"):
    return {"id": f"c{i}", "text": f"{text} {i}", "parent_id": pid, "source_file": "a.pdf", "page": 1}


def test_replaces_child_with_parent_and_keeps_snippet():
    parents = {"p1": {"text": "toàn bộ section một " * 10}}
    out = expand_chunks_with_parents([child(1, "p1")], parents, 1000, ct)
    assert out[0]["expanded"] is True
    assert out[0]["text"].startswith("toàn bộ section")
    assert out[0]["snippet"] == "child text 1"
    assert out[0]["child_text"] == "child text 1"


def test_dedupes_children_of_same_parent_keeps_rank_order():
    parents = {"p1": {"text": "P1 " * 5}, "p2": {"text": "P2 " * 5}}
    chunks = [child(1, "p1"), child(2, "p2"), child(3, "p1")]
    out = expand_chunks_with_parents(chunks, parents, 1000, ct)
    assert [c["parent_id"] for c in out] == ["p1", "p2"]


def test_budget_falls_back_to_child_when_parent_too_big():
    parents = {"p1": {"text": "x " * 100}, "p2": {"text": "y " * 5}}
    out = expand_chunks_with_parents([child(1, "p1"), child(2, "p2")], parents, 50, ct)
    assert not out[0].get("expanded") and out[0]["text"] == "child text 1"
    assert out[1].get("expanded")


def test_missing_parent_or_no_parent_id_passes_through():
    out = expand_chunks_with_parents([child(1, "ghost"), {"text": "no parent"}], {}, 1000, ct)
    assert len(out) == 2 and all("snippet" in c for c in out)
    assert out[1]["text"] == "no parent"


def test_does_not_mutate_input():
    chunks = [child(1, "p1")]
    expand_chunks_with_parents(chunks, {"p1": {"text": "P " * 5}}, 1000, ct)
    assert "expanded" not in chunks[0] and chunks[0]["text"] == "child text 1"


def test_snippet_truncates_and_collapses_whitespace():
    s = make_snippet("a   b\n\nc " + "x" * 400, limit=50)
    assert len(s) <= 50 and s.endswith("…") and "  " not in s
