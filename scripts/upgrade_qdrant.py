"""
Nâng cấp Qdrant server (docker-compose) TỪNG MINOR MỘT — yêu cầu của Qdrant khi
storage đã có dữ liệu — tới phiên bản khớp qdrant-client trong requirements.txt.

    python scripts/upgrade_qdrant.py            # chạy thật
    python scripts/upgrade_qdrant.py --dry-run  # chỉ in kế hoạch

An toàn:
1. Trước khi nâng: tạo snapshot mọi collection và TẢI VỀ ./backups/qdrant/<thời điểm>/
   (dữ liệu vector không tái tạo được vì file gốc không được lưu lại).
2. Sau mỗi bước: chờ server sẵn sàng, kiểm tra phiên bản và số điểm từng collection
   không đổi; lệch => dừng ngay, giữ nguyên phiên bản đang chạy.
3. Xong: cập nhật tag image trong docker-compose.yml.

Khôi phục từ snapshot (nếu cần): xem https://qdrant.tech/documentation/concepts/snapshots/
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPOSE_FILE = os.path.join(ROOT, "docker-compose.yml")
QDRANT_URL = "http://127.0.0.1:6333"

# Bản vá mới nhất của từng minor (Docker Hub, 2026-10-31)
UPGRADE_PATH = [
    "v1.11.5", "v1.12.6", "v1.13.6", "v1.14.1", "v1.15.5",
    "v1.16.3", "v1.17.1", "v1.18.3", "v1.19.2",
]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def _http(method: str, path: str, body: dict | None = None, timeout: int = 120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        QDRANT_URL + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def server_version() -> str | None:
    try:
        return "v" + _http("GET", "/", timeout=3)["version"]
    except Exception:
        return None


def wait_ready(expected: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_version() == expected:
            try:
                _http("GET", "/collections", timeout=3)
                return
            except Exception:
                pass
        time.sleep(2)
    raise SystemExit(f"Qdrant {expected} không sẵn sàng sau {timeout_s}s — dừng lại.")


def collection_counts() -> dict[str, int]:
    names = [c["name"] for c in _http("GET", "/collections")["result"]["collections"]]
    return {n: _http("POST", f"/collections/{n}/points/count", {"exact": True})["result"]["count"] for n in names}


def backup_snapshots(counts: dict[str, int]) -> str:
    out_dir = os.path.join(ROOT, "backups", "qdrant", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    for name in counts:
        snap = _http("POST", f"/collections/{name}/snapshots?wait=true", timeout=1800)["result"]["name"]
        dest = os.path.join(out_dir, f"{name}__{snap}")
        urllib.request.urlretrieve(f"{QDRANT_URL}/collections/{name}/snapshots/{snap}", dest)
        print(f"  ✔ {name}: {counts[name]} điểm -> {dest} ({os.path.getsize(dest) // 1024} KB)")
    with open(os.path.join(out_dir, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    return out_dir


def set_compose_image(tag: str) -> None:
    with open(COMPOSE_FILE, encoding="utf-8") as f:
        text = f.read()
    new_text, n = re.subn(r"image: qdrant/qdrant:v[\d.]+", f"image: qdrant/qdrant:{tag}", text)
    if n != 1:
        raise SystemExit("Không tìm thấy dòng 'image: qdrant/qdrant:...' trong docker-compose.yml.")
    with open(COMPOSE_FILE, "w", encoding="utf-8") as f:
        f.write(new_text)


def run_qdrant(tag: str) -> None:
    set_compose_image(tag)
    subprocess.run(["docker", "compose", "up", "-d", "qdrant"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    current = server_version()
    if current is None:
        raise SystemExit("Không kết nối được Qdrant ở 127.0.0.1:6333. Chạy: docker compose up -d qdrant")
    if current not in UPGRADE_PATH:
        raise SystemExit(f"Phiên bản hiện tại {current} không nằm trong lộ trình {UPGRADE_PATH}.")

    steps = UPGRADE_PATH[UPGRADE_PATH.index(current) + 1:]
    print(f"Qdrant hiện tại: {current}")
    if not steps:
        print("Đã ở phiên bản đích, không cần nâng cấp.")
        return
    print("Lộ trình: " + " -> ".join([current, *steps]))

    counts = collection_counts()
    print(f"Collections: {counts or '(trống)'}")
    if args.dry_run:
        return

    if counts:
        print("Sao lưu snapshot trước khi nâng cấp...")
        print(f"Đã lưu tại: {backup_snapshots(counts)}")

    for tag in steps:
        print(f"\n→ Nâng lên {tag} ...")
        run_qdrant(tag)
        wait_ready(tag)
        after = collection_counts()
        if after != counts:
            raise SystemExit(f"Số điểm thay đổi sau khi lên {tag}: {counts} -> {after}. DỪNG — kiểm tra trước khi tiếp tục.")
        print(f"  ✔ {tag} OK, dữ liệu nguyên vẹn: {after}")

    print(f"\nHoàn tất: Qdrant {steps[-1]}; docker-compose.yml đã được cập nhật.")


if __name__ == "__main__":
    main()
