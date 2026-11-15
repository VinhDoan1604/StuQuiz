"""
scripts/e2e_smoke.py

Kiểm tra end-to-end trên stack ĐANG CHẠY (qua nginx, gọi LLM thật): đăng ký + OTP, upload,
hỏi đáp qua SSE có trích dẫn, từ chối khi tài liệu không có, câu nối tiếp, giảng bài, quiz chấm
phía server, mastery, chặn truy cập chéo giữa hai user. Cần APP_ENV=dev để nhận mã OTP.

    python scripts/e2e_smoke.py --file eval/corpus/wiki_dijkstra.md --out e2e.json

Tài liệu và không gian tạo ra được xoá ở cuối; hai tài khoản e2e-*@example.test thì còn lại.
"""
from __future__ import annotations

import argparse
import json
import secrets
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

API = "http://localhost:8080/api"
TIMEOUT = 240


class Client:
    def __init__(self, label: str) -> None:
        self.http = httpx.Client(base_url=API, timeout=TIMEOUT)
        email = f"e2e-{label}-{secrets.token_hex(4)}@example.test"
        password = secrets.token_urlsafe(16)
        reg = self.http.post("/auth/register", json={"email": email, "password": password, "full_name": f"E2E {label}"})
        reg.raise_for_status()
        code = reg.json().get("dev_code")
        if not code:
            raise SystemExit("Không nhận được dev_code: stack phải chạy với APP_ENV=dev.")
        ok = self.http.post("/auth/verify-otp", json={"email": email, "code": code})
        ok.raise_for_status()
        self.http.headers["Authorization"] = f"Bearer {ok.json()['token']}"

    def chat(self, session: str, message: str, doc_ids: list[str], space_id: str) -> dict[str, Any]:
        """Một lượt SSE: trả về response cuối, các node đã đi qua và độ trễ tới token đầu/kết thúc."""
        body = {"session_id": session, "message": message, "document_ids": doc_ids, "space_id": space_id}
        start, first_token, nodes, tokens, response = time.perf_counter(), None, [], 0, None
        with self.http.stream("POST", "/chat/stream", json=body) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                kind = event.get("event")
                if kind == "step":
                    nodes.append(event["node"])
                elif kind == "token":
                    tokens += 1
                    first_token = first_token or time.perf_counter() - start
                elif kind == "done":
                    response = event["response"]
                elif kind == "error":
                    raise RuntimeError(event.get("message"))
        if response is None:
            raise RuntimeError("Stream kết thúc mà không có sự kiện done.")
        return {"response": response, "nodes": nodes, "token_events": tokens,
                "first_token_s": first_token, "total_s": time.perf_counter() - start}


def wait_ready(client: Client, doc_id: str) -> dict[str, Any]:
    for _ in range(120):
        status = client.http.get(f"/documents/{doc_id}/status").json()
        if status.get("status") in ("ready", "failed", "error"):
            return status
        time.sleep(2)
    raise TimeoutError("Hết giờ chờ index tài liệu.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: Any = "") -> None:
        checks.append((name, bool(passed), str(detail)[:200]))
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {str(detail)[:200]}", flush=True)

    a, b = Client("a"), Client("b")
    path = Path(args.file)

    t0 = time.perf_counter()
    upload = a.http.post("/documents/upload", files={"file": (path.name, path.read_bytes())})
    upload.raise_for_status()
    doc_id = upload.json().get("document_id") or upload.json().get("id")
    status = wait_ready(a, doc_id)
    check("upload + index", status.get("status") == "ready", f"{status} in {time.perf_counter() - t0:.0f}s")

    space = a.http.post("/spaces", json={"title": "E2E", "description": "smoke"}).json()
    space_id = space["id"]
    a.http.post(f"/spaces/{space_id}/documents", json={"document_id": doc_id}).raise_for_status()
    session = f"e2e-{secrets.token_hex(4)}"
    timings: list[dict[str, Any]] = []

    qa = a.chat(session, "Thuật toán Dijkstra dùng để giải bài toán gì?", [doc_id], space_id)
    timings.append(qa)
    check("QA streams tokens", qa["token_events"] > 0, f"{qa['token_events']} token events")
    check("QA has citations", bool(qa["response"]["citations"]), qa["response"]["citations"][:1])
    check("QA route skips reflection", not {"grade_context", "grade_answer"} & set(qa["nodes"]), qa["nodes"])
    check("QA turn saved", bool(qa["response"].get("turn_id")))

    follow = a.chat(session, "Độ phức tạp của nó khi dùng hàng đợi ưu tiên là bao nhiêu?", [doc_id], space_id)
    timings.append(follow)
    rewritten = follow["response"]["reflection"]["rewritten_queries"]
    check("Follow-up rewritten", any("dijkstra" in q.lower() for q in rewritten), rewritten)

    refuse = a.chat(f"{session}-r", "Thủ đô của nước Pháp có dân số bao nhiêu người vào năm 2020?", [doc_id], space_id)
    timings.append(refuse)
    answer = refuse["response"]["answer"].lower()
    check("Refuses out-of-scope", any(m in answer for m in ("không đề cập", "không có thông tin", "không cung cấp", "không chứa")), answer[:120])

    teach = a.chat(f"{session}-t", "Giảng cho mình hiểu từng bước thuật toán Dijkstra hoạt động thế nào", [doc_id], space_id)
    timings.append(teach)
    check("Teaching intent routed", teach["response"]["intent"] == "teaching", teach["response"]["intent"])

    if qa["response"].get("turn_id"):
        fb = a.http.post(f"/chat/turns/{qa['response']['turn_id']}/feedback", json={"rating": 1})
        check("Feedback stored", fb.status_code == 200, fb.status_code)

    quiz = a.http.post("/quiz/generate", json={"document_ids": [doc_id], "num_questions": 2, "space_id": space_id})
    check("Quiz generated", quiz.status_code == 200 and len(quiz.json()) >= 1, quiz.status_code)
    questions = quiz.json() if quiz.status_code == 200 else []
    check("Quiz hides answers", all(not ({"correct_choice", "explanation", "answer"} & set(q)) for q in questions), [sorted(q) for q in questions[:1]])
    for q in questions:
        graded = a.http.post("/quiz/answer", json={"attempt_id": q["attempt_id"], "choice": q["choices"][0]})
        check("Quiz graded by server", graded.status_code == 200 and "is_correct" in graded.json(), graded.status_code)
    mastery = a.http.get("/mastery/all").json()
    check("Mastery updated", len(mastery) >= 1, mastery[:1])
    if questions:
        again = b.http.post("/quiz/answer", json={"attempt_id": questions[0]["attempt_id"], "choice": "x"})
        check("Isolation: other user's quiz attempt", again.status_code in (403, 404), again.status_code)

    check("Isolation: document status", b.http.get(f"/documents/{doc_id}/status").status_code == 404)
    check("Isolation: space", b.http.get(f"/spaces/{space_id}").status_code == 404)
    check("Isolation: delete document", b.http.delete(f"/documents/{doc_id}").status_code == 404)
    # Chat cố ý bỏ qua id không thuộc user (strict=False, id cũ đã xoá không được làm hỏng hội thoại),
    # nên 200 là đúng; điều phải đảm bảo là không có đoạn nào của user A lọt vào câu trả lời.
    probe = b.http.post("/chat", json={"session_id": "x", "message": "Dijkstra là gì?", "document_ids": [doc_id]})
    check("Isolation: chat cannot read another user's document", probe.status_code == 200 and probe.json()["citations"] == [], probe.text[:150])

    report = {
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "first_token_s": [round(t["first_token_s"], 1) for t in timings if t["first_token_s"]],
        "total_s": [round(t["total_s"], 1) for t in timings],
        "median_total_s": round(statistics.median(t["total_s"] for t in timings), 1),
    }

    a.http.delete(f"/spaces/{space_id}")
    a.http.delete(f"/documents/{doc_id}")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(not p for _, p, _ in checks)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
