"""
eval/build_v2_drafts.py — soạn nháp câu hỏi cho dataset v2 rồi KIỂM TRA bằng code.

Nhắm vào 3 điểm yếu của dataset v1 (xem eval/FOLLOWUP.md):
  1. Câu không có đáp án quá dễ (từ chối đúng = 1.0 ở mọi cấu hình) -> thêm câu
     "hard-unanswerable": chủ đề NẰM TRONG corpus nhưng chi tiết được hỏi thì không có
     (năm/tạp chí công bố, con số cụ thể, cơ chế chỉ được nhắc tên).
  2. Câu hỏi sinh tự động trùng từ vựng với tài liệu -> thêm câu "student-phrasing":
     viết tắt, không dấu, văn nói, sai chính tả nhẹ, trộn Anh–Việt.
  3. n nhỏ -> v1 (48) + 33 câu nháp = 81 câu (58 có đáp án, 23 không có đáp án).

Mọi câu nháp có reviewed=false: load_dataset() BỎ QUA chúng cho tới khi người duyệt
đổi thành true (hoặc chạy eval với --include-unreviewed).

Kiểm tra tự động (cần Qdrant đã nạp corpus):
  - answerable: mỗi gold_evidence khớp >=1 chunk (cùng hàm với preflight của eval/run.py)
    VÀ chunk khớp thuộc đúng source_file;
  - unanswerable: không chunk nào chứa các cụm `absent` (bằng chứng phụ cho nhãn
    "không có đáp án" — người duyệt vẫn phải đọc lại).

    python -m eval.build_v2_drafts            # kiểm tra + ghi eval/data/questions_v2.jsonl
    python -m eval.build_v2_drafts --check    # chỉ kiểm tra
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from eval.metrics import evidence_match, normalize_text, split_alternatives

V1_PATH = Path("eval/data/questions.jsonl")
OUT_PATH = Path("eval/data/questions_v2.jsonl")
MANIFEST = Path("eval/data/corpus_manifest.json")

S = "student-phrasing"
H = "hard-unanswerable"

ANSWERABLE = [
    dict(id="v2s01", q="cho em hỏi query trong DBMS được xử lý qua mấy bước v ạ, bước optimizer thì làm gì",
         a="5 bước: parsing, translation, optimizer, execution, evaluation. Optimizer chọn kế hoạch thực thi tốt nhất (chiến lược chi phí thấp nhất).",
         ev=["Processing the query includes five steps", "optimizer: selecting the best plan to execute"],
         src="openstax_intro_cs_ch8_data_management.pdf", tags=["lang-vi", S, "cross-lingual", "procedure"]),
    dict(id="v2s02", q="phép chia (division) trong đại số quan hệ viết thẳng bằng SQL đc ko",
         a="Không, toán tử Division không được cài đặt trực tiếp trong SQL.",
         ev=["the Division operator is not directly implemented in SQL"],
         src="openstax_intro_cs_ch8_data_management.pdf", tags=["lang-vi", S, "cross-lingual", "fact"]),
    dict(id="v2s03", q="5 phép toán nguyên thủy của relational algebra là mấy cái nào thế",
         a="Select, Project, Difference, Product và Union; các phép khác đều suy ra được từ 5 phép này.",
         ev=["are often referred to as the five primitive operators because all other relational algebra operators can be derived from them"],
         src="openstax_intro_cs_ch8_data_management.pdf", tags=["lang-vi", S, "cross-lingual", "list"]),
    dict(id="v2s04", q="binary search muốn chạy O(log N) thì cần đk gì",
         a="Danh sách phải được sắp xếp và việc truy cập theo chỉ số phải là thao tác thời gian hằng số.",
         ev=["so long as indexing is a constant-time operation"],
         src="openstax_intro_cs_ch3_data_structures_algorithms.pdf", tags=["lang-vi", S, "cross-lingual", "condition"]),
    dict(id="v2s05", q="simulation trong DSS có chắc ra nghiệm tối ưu ko hay chỉ tương đối",
         a="Không đảm bảo nghiệm tối ưu, thường chỉ tìm được nghiệm tương đối tốt.",
         ev=["Không đảm bảo nghiệm tối ưu"],
         src="notes_co4031_data_warehouse_dss.pdf", tags=["lang-vi", S, "fact"]),
    dict(id="v2s06", q="hồi băng hà cỡ 18 nghìn năm trước nước biển xuống thấp cỡ nào so vs bây giờ",
         a="Mực nước biển dao động mạnh, có lúc thấp hơn ngày nay tới khoảng 120 m.",
         ev=["Mực nước biển dao động mạnh có những lúc xuống đến 120m thấp hơn so với ngày nay"],
         src="voer_lich_su_tu_nhien_viet_nam.md", tags=["lang-vi", S, "number"]),
    dict(id="v2s07", q="A* ai nghĩ ra z, năm nào",
         a="Peter Hart, Nils Nilsson và Bertram Raphael mô tả lần đầu năm 1968.",
         ev=["Thuật toán A* được mô tả lần đầu vào năm 1968 bởi Peter Hart, Nils Nilsson, và Bertram Raphael"],
         src="wiki_a_star.md", tags=["lang-vi", S, "fact"]),
    dict(id="v2s08", q="lam sao dung bfs de kiem tra do thi hai phia",
         a="BFS từ một đỉnh bất kỳ, gán nhãn chẵn/lẻ (0 cho gốc, 1 cho đỉnh kề, ...); nếu có hai đỉnh kề nhau cùng nhãn thì không phải đồ thị hai phía, ngược lại là hai phía.",
         ev=["gán nhãn 0 cho đỉnh gốc, 1 cho tất cả các đỉnh kề đỉnh gốc",
             "có hai đỉnh kề nhau có cùng nhãn, thì đồ thị không là hai phía"],
         src="wiki_bfs.md", tags=["lang-vi", S, "no-diacritics", "procedure"]),
    dict(id="v2s09", q="bst mà ko tự cân bằng thì chèn xoá nhiều bị sao ạ",
         a="Cây có thể suy biến, độ cao tới n, và hiệu suất truy vấn giảm xuống tương đương tìm kiếm tuyến tính.",
         ev=["hiệu suất truy vấn giảm xuống còn tương đương với tìm kiếm tuyến tính"],
         src="wiki_binary_search_tree.md", tags=["lang-vi", S, "why"]),
    dict(id="v2s10", q="kruskal độ phức tạp bn vậy mn",
         a="O(E log V) (sắp xếp cạnh O(E log E) cộng cấu trúc tập rời nhau).",
         ev=["thuật toán Kruskal chạy trong thời gian O(E log V)"],
         src="wiki_kruskal.md", tags=["lang-vi", S, "number"]),
    dict(id="v2s11", q="muốn tìm cây khung lớn nhất thì làm sao, có dùng lại thuật toán cây khung nhỏ nhất đc ko",
         a="Được: nhân mọi trọng số với -1 rồi giải bài toán cây bao trùm nhỏ nhất trên đồ thị mới.",
         ev=["nhân các trọng số với -1 và giải bài toán cây bao trùm nhỏ nhất trên đồ thị mới"],
         src="wiki_minimum_spanning_tree.md", tags=["lang-vi", S, "paraphrased", "procedure"]),
    dict(id="v2s12", q="khi nào thì 1 đồ thị có thứ tự topo",
         a="Khi và chỉ khi đồ thị có hướng không có chu trình (DAG).",
         ev=["Thứ tự tô pô tồn tại khi và chỉ khi đồ thị không có chu trình"],
         src="wiki_topological_sort.md", tags=["lang-vi", S, "condition"]),
    dict(id="v2s13", q="dn muốn xin giấy phép mua bán vàng miếng thì vốn điều lệ tối thiểu bao nhiêu",
         a="Từ 100 tỷ đồng trở lên (kèm các điều kiện khác như kinh nghiệm 2 năm, thuế đã nộp, mạng lưới 3 tỉnh).",
         ev=["Có vốn điều lệ từ 100 tỷ đồng trở lên"],
         src="law_nd24_2012_kinh_doanh_vang_vbhn117_2026.md", tags=["lang-vi", S, "number", "law", "disambiguation"]),
    dict(id="v2s14", q="nhnn dựa vào đâu để chia hạn mức xuất nhập khẩu vàng cho từng dn",
         a="Trên cơ sở tổng hạn mức hàng năm, NHNN cấp hạn mức cho doanh nghiệp, ngân hàng thương mại theo quy mô vốn điều lệ (và tình hình xuất khẩu, nhập khẩu).",
         ev=["cấp hạn mức hàng năm cho doanh nghiệp, ngân hàng thương mại theo quy mô vốn điều lệ"],
         src="law_nd24_2012_kinh_doanh_vang_vbhn117_2026.md", tags=["lang-vi", S, "law"]),
    dict(id="v2s15", q="prim mà xài fibonacci heap thì nhanh cỡ nào",
         a="Thời gian chạy giảm xuống O(E + V log V).",
         ev=["có thể giảm thời gian chạy xuống O(E + V log V)"],
         src="wiki_prim.md", tags=["lang-vi", S, "number"]),
    dict(id="v2s16", q="cuckoo hashing có gì hay hơn mấy cách giải quyết va chạm khác",
         a="Đảm bảo thời gian tìm kiếm hằng số trong trường hợp xấu nhất, chèn và xoá có thời gian trừ dần hằng số.",
         ev=["đảm bảo thời gian cho mỗi phép tìm kiếm là hằng số trong trường hợp xấu nhất"],
         src="wiki_hash_table.md", tags=["lang-vi", S, "mechanism"]),
    dict(id="v2s17", q="EAC tính sao vậy ạ, có vd ko",
         a="EAC = BAC/CPI; ví dụ BAC £19,000, CPI 1.86 thì EAC ≈ £10,215.",
         ev=["Estimate at completion (EAC) – updated estimate = BAC/CPI"],
         src="slides_spm_ch9_monitoring_and_control.pptx", tags=["lang-vi", S, "cross-lingual", "number"]),
    dict(id="v2s18", q="whats the diff between best case n worst case growth for sequential search",
         a="Best case: constant order of growth; worst case: linear order of growth with respect to N.",
         ev=["In the best case, the sequential search has a constant order of growth",
             "In the worst case, the sequential search has a linear order of growth"],
         src="openstax_intro_cs_ch3_data_structures_algorithms.pdf", tags=["lang-en", S, "en-same-lang", "compare"]),
    dict(id="v2s19", q="what r some example apps of an analytics process model",
         a="Risk analytics (credit scoring, fraud detection), marketing analytics, response modeling, customer segmentation, recommender systems.",
         ev=["risk analytics (e.g., credit scoring, fraud detection), marketing analytics"],
         src="openstax_intro_cs_ch8_data_management.pdf", tags=["lang-en", S, "en-same-lang", "list"]),
    dict(id="v2s21", q="ở vn có mấy loài thằn lằn rắn v, sao lại có họ hàng tận bắc mỹ",
         a="3 loài; chúng có cả họ hàng gần và tổ tiên xa ở Bắc Mỹ do một nhóm từ Đông Á phát tán ngược lại Bắc Mỹ qua cầu nối đất liền Bering.",
         ev=["3 loài thằn lằn rắn của Việt Nam có cả họ hàng gần gũi và tổ tiên xa ở Bắc Mỹ"],
         src="voer_lich_su_tu_nhien_viet_nam.md", tags=["lang-vi", S, "number", "why"]),
    dict(id="v2s20", q="cây AVL ai tạo ra vs năm nào",
         a="Georgy Adelson-Velsky và Evgenii Landis, năm 1962.",
         ev=["Cây AVL là cây tìm kiếm nhị phân tự cân bằng đầu tiên do Georgy Adelson-Velsky và Evgenii Landis cho ra đời năm 1962"],
         src="wiki_binary_search_tree.md", tags=["lang-vi", S, "fact"]),
]

# absent: các cụm (regex, không phân biệt hoa thường) mà NẾU có trong corpus thì nhãn
# "không có đáp án" đáng ngờ. Mỗi câu ghi chú vì sao "khó" (ngay sát nội dung có thật).
UNANSWERABLE = [
    dict(id="v2u01", q="Thuật toán A* được công bố năm 1968 trên tạp chí khoa học nào?",
         absent=[r"IEEE", r"Transactions", r"tạp chí", r"journal"], files="wiki_a_star",
         note="Tài liệu có năm và tác giả (chunk A*), không có nơi công bố.",
         tags=["lang-vi", H, "near-miss", "graph"]),
    dict(id="v2u02", q="Query optimizer dùng histogram thống kê để ước lượng chi phí kế hoạch thực thi như thế nào?",
         absent=[r"histogram"],
         note="Tài liệu nói optimizer chọn chiến lược chi phí thấp nhất + heuristic, không nói histogram.",
         tags=["lang-vi", H, "near-miss", "cross-lingual"]),
    dict(id="v2u03", q="Slowly changing dimension loại 2 trong kho dữ liệu lưu lịch sử thay đổi của chiều ra sao?",
         absent=[r"slowly changing", r"\bSCD\b", r"thay đổi chậm"],
         note="Ghi chú DW rất dài (star/snowflake, Kimball...) nhưng không có SCD.",
         tags=["lang-vi", H, "out-of-scope", "data-warehouse"]),
    dict(id="v2u04", q="Ngân hàng Nhà nước cấp tổng hạn mức nhập khẩu vàng năm 2025 là bao nhiêu tấn?",
         absent=[r"tấn"],
         note="Tài liệu mô tả CÁCH xác định tổng hạn mức, không có con số.",
         tags=["lang-vi", H, "near-miss", "law", "number"]),
    dict(id="v2u05", q="Việt Nam có bao nhiêu loài ba ba thuộc họ Trionychidae?",
         absent=[r"\d+ loài ba ba", r"Trionychidae[^.]{0,80}\d+ loài", r"(ba|hai|bốn|năm) loài ba ba"],
         note="Ngay cạnh ví dụ thằn lằn rắn (CÓ số loài: 3) tài liệu nhắc ba ba họ Trionychidae, nhưng không có số loài ba ba.",
         tags=["lang-vi", H, "near-miss", "biology", "number"]),
    dict(id="v2u06", q="Thuật toán Bellman–Ford được công bố lần đầu vào năm nào?",
         absent=[r"năm (19|20)\d\d", r"\((19|20)\d\d\)"], files="wiki_bellman_ford",
         note="Bài Bellman–Ford có chứng minh, độ phức tạp nhưng không có năm.",
         tags=["lang-vi", H, "near-miss", "graph"]),
    dict(id="v2u07", q="Gantt chart được dùng thế nào để theo dõi tiến độ dự án trong chương monitoring and control?",
         absent=[r"Gantt"],
         note="Slide có earned value, SPI/CPI nhưng không có Gantt chart.",
         tags=["lang-vi", H, "near-miss", "project-management", "cross-lingual"]),
    dict(id="v2u08", q="How does Tarjan's algorithm find the strongly connected components of a directed graph?",
         absent=[r"Tarjan[^.]{0,80}(liên thông mạnh|strongly)"],
         note="Bài DFS chỉ liệt kê 'xác định thành phần liên thông mạnh' là ứng dụng; Tarjan chỉ xuất hiện ở sắp xếp tô pô/MST.",
         tags=["lang-en", H, "near-miss", "graph"]),
    dict(id="v2u09", q="How is a B+ tree leaf node split when an insertion makes it overflow?",
         absent=[r"B\+[^.]{0,100}(split|tách|overflow)", r"(split|tách)[^.]{0,60}(leaf|nút lá)"],
         note="B+ tree chỉ được nhắc tên trong phần tổ chức file / chỉ mục.",
         tags=["lang-en", H, "near-miss", "cross-lingual"]),
    dict(id="v2u10", q="Nghị định 24/2012 quy định mức phạt bao nhiêu khi doanh nghiệp không niêm yết giá mua, bán vàng miếng?",
         absent=[r"phạt tiền", r"triệu đồng đến"],
         note="Tài liệu có NGHĨA VỤ niêm yết giá nhưng không có khung phạt tiền.",
         tags=["lang-vi", H, "near-miss", "law", "number"]),
    dict(id="v2u11", q="dijkstra mà gặp cạnh âm thì sửa code kiểu gì để vẫn chạy đúng v ạ",
         absent=[r"sửa[^.]{0,40}Dijkstra[^.]{0,40}âm"],
         note="Văn phong sinh viên + hỏi cách 'sửa' Dijkstra: tài liệu chỉ nói trọng số phải không âm (và có Bellman–Ford), không có cách sửa Dijkstra.",
         tags=["lang-vi", H, S, "near-miss", "graph"]),
    dict(id="v2u12", q="cho e hỏi đề thi HK232 môn CO4031 câu 55 đáp án là gì",
         absent=[r"^\s*5[0-9]\s*\|"],
         note="Bảng đáp án HK232 trong ghi chú chỉ tới khoảng câu 42.",
         tags=["lang-vi", H, S, "near-miss", "data-warehouse"]),
]


def build_items() -> list[dict]:
    items = []
    for d in ANSWERABLE:
        items.append({
            "id": d["id"], "question": d["q"], "answerable": True, "gold_answer": d["a"],
            "gold_evidence": d["ev"], "source_file": d["src"], "tags": d["tags"] + ["claude-drafted", "v2"],
            "reviewed": False,
        })
    for d in UNANSWERABLE:
        items.append({
            "id": d["id"], "question": d["q"], "answerable": False, "gold_answer": None,
            "gold_evidence": [], "tags": ["unanswerable"] + d["tags"] + ["claude-drafted", "v2"],
            "review_note": d["note"], "reviewed": False,
        })
    return items


async def _load_chunks() -> list[dict]:
    from app.retrieval.helpers import aget_all_chunks_for_documents

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chunks = await aget_all_chunks_for_documents([d["doc_id"] for d in manifest["documents"]])
    if not chunks:
        raise SystemExit("Index rỗng — chạy `python -m eval.ingest_corpus` trước.")
    return chunks


def check(chunks: list[dict]) -> list[str]:
    norm = [(normalize_text(c["text"]), c.get("source_file")) for c in chunks]
    problems = []
    for d in ANSWERABLE:
        for ev in d["ev"]:
            for alt in split_alternatives(ev):
                files = {f for t, f in norm if evidence_match(t, alt, normalized=True)}
                if not files:
                    problems.append(f"{d['id']}: evidence không khớp chunk nào: {alt[:70]!r}")
                elif d["src"] not in files:
                    problems.append(f"{d['id']}: evidence khớp file khác {sorted(files)}")
    for d in UNANSWERABLE:
        scope = [c for c in chunks if (c.get("source_file") or "").startswith(d.get("files", ""))]
        for pat in d["absent"]:
            hits = [c.get("source_file") for c in scope if re.search(pat, c["text"], re.I | re.M)]
            if hits:
                problems.append(f"{d['id']}: cụm /{pat}/ CÓ trong corpus ({hits[:3]}) — xem lại nhãn")
    v1_ids = set()
    for line in V1_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            v1_ids.add(json.loads(line)["id"])
    dup = v1_ids & {i["id"] for i in build_items()}
    if dup:
        problems.append(f"id trùng với v1: {sorted(dup)}")
    return problems


def write(items: list[dict]) -> None:
    lines = [
        "# Dataset v2 = v1 (48 câu đã duyệt) + 33 câu nháp (reviewed=false) do eval/build_v2_drafts.py sinh.",
        "# Câu nháp bị load_dataset() bỏ qua cho tới khi người duyệt đổi reviewed=true. Xem eval/FOLLOWUP.md.",
    ]
    for line in V1_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            lines.append(line)
    lines += [json.dumps(i, ensure_ascii=False) for i in items]
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Đã ghi {OUT_PATH} ({len(lines) - 2} câu)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi file")
    args = ap.parse_args()
    problems = check(asyncio.run(_load_chunks()))
    for p in problems:
        print("[lỗi] " + p)
    n_a, n_u = len(ANSWERABLE), len(UNANSWERABLE)
    print(f"{n_a} câu có đáp án, {n_u} câu không có đáp án; {len(problems)} vấn đề")
    if problems:
        raise SystemExit(1)
    if not args.check:
        write(build_items())


if __name__ == "__main__":
    main()
