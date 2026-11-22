"""eval/ocr_eval.py: báo cáo CER/WER và hiệu paired so với Tesseract (hàm thuần, không gọi engine)."""
import json

from eval.ocr_eval import build_report, merge_runs, score

META = {"run_id": "t", "ocr_lang": "vie+eng", "git_commit": "abc"}


def _rec(item, level, engine, cer_value, error=None):
    return {"item": item, "level": level, "engine": engine, "cer": cer_value, "cer_nodiac": cer_value / 2,
            "wer": cer_value * 2, "seconds": 1.0, "tokens_in": 0 if engine == "tesseract" else 1000,
            "tokens_out": 0 if engine == "tesseract" else 200, "fallbacks": 0, "error": error}


def _records():
    items = [("a", "light"), ("b", "medium"), ("c", "heavy"), ("d", "light")]
    rows = [_rec(i, lv, "tesseract", 0.20 + k * 0.01) for k, (i, lv) in enumerate(items)]
    rows += [_rec(i, lv, "vlm:m", 0.02) for i, lv in items]
    return rows


def test_score_is_zero_for_identical_text_up_to_whitespace():
    assert score("Kho dữ liệu\nlà gì", "Kho dữ liệu là gì") == {"cer": 0.0, "cer_nodiac": 0.0, "wer": 0.0}


def test_report_puts_tesseract_first_and_flags_clear_paired_gain():
    summary, md = build_report(list(reversed(_records())), META, n_boot=500)
    assert list(summary["engines"]) == ["tesseract", "vlm:m"]
    d = summary["paired_vs_tesseract"]["vlm:m"]["cer"]
    assert d["n"] == 4 and d["hi"] < 0
    assert "fewer errors" in md
    assert summary["engines"]["vlm:m"]["tokens_mean"] == 1200


def test_failed_calls_are_counted_and_left_out_of_the_paired_diff():
    rows = _records()
    rows[-1] = _rec("d", "light", "vlm:m", 1.0, error="RuntimeError: 503")
    summary, md = build_report(rows, META, n_boot=200)
    assert summary["engines"]["vlm:m"]["errors"] == 1
    assert summary["paired_vs_tesseract"]["vlm:m"]["cer"]["n"] == 3
    assert "| 1/4 |" in md


def test_identical_engines_report_no_demonstrated_difference():
    rows = [_rec(i, "light", e, 0.1) for i in "abc" for e in ("tesseract", "vlm:m")]
    _, md = build_report(rows, META, n_boot=200)
    assert "no demonstrated difference" in md


def test_merge_runs_combines_engines_without_duplicates(tmp_path):
    recs = _records()
    for name, engine in (("r1", None), ("r2", "vlm:m")):
        src = tmp_path / name
        src.mkdir()
        rows = recs if engine is None else [r for r in recs if r["engine"] == engine]
        (src / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (src / "meta.json").write_text(json.dumps({**META, "run_id": name, "dataset": "x", "degradation": {},
                                                   "metrics": "m"}), encoding="utf-8")
    out = tmp_path / "merged"
    out.mkdir()
    merge_runs([tmp_path / "r1", tmp_path / "r2"], out, "merged", n_boot=200)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["engines"]["vlm:m"]["n"] == 4  # bản trùng ở r2 bị bỏ
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["merged_from"] == ["r1", "r2"]


def test_rescore_uses_current_ground_truth_and_labels_old_vlm_runs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.gt.txt").write_text("đúng rồi", encoding="utf-8")
    src = tmp_path / "old"
    src.mkdir()
    rows = [{**_rec("a.jpg", "light", "vlm", 0.9), "text": "đúng rồi"},
            {**_rec("a.jpg", "light", "tesseract", 0.9), "text": "dung roi"}]
    (src / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (src / "meta.json").write_text(json.dumps({**META, "run_id": "old", "dataset": "x", "degradation": {},
                                               "metrics": "m", "vlm_model": "p:gem"}), encoding="utf-8")
    out = tmp_path / "new"
    out.mkdir()
    merge_runs([src], out, "new", n_boot=100, rescore_data=data)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["engines"]["vlm:gem"]["cer"]["mean"] == 0.0
    assert summary["engines"]["tesseract"]["cer"]["mean"] > 0
