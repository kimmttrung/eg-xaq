"""Tải điểm cháy NASA FIRMS cho đúng các ngày trong bộ episode.

    python scripts/download_firms.py
    python scripts/download_firms.py --episodes data/eval/episodes.yaml --source VIIRS_SNPP_SP

Cần FIRMS_MAP_KEY trong .env — đăng ký miễn phí tại
https://firms.modaps.eosdis.nasa.gov/api/map_key/

Chỉ tải NGÀY CẦN (mỗi episode: ngày đó + hôm trước), không tải cả ba năm: vài chục
request thay vì vài nghìn, và nằm xa dưới hạn mức của API.

Ghi hai file:
    data/raw/firms_hanoi.csv          các điểm cháy (latitude, longitude, acq_date, frp, ...)
    data/raw/firms_hanoi.csv.dates    danh sách ngày ĐÃ TRA THÀNH CÔNG

File thứ hai là bắt buộc: ngày không có điểm cháy nào sẽ không có dòng nào trong CSV,
nên chỉ nhìn CSV thì không phân biệt được "không có cháy" với "chưa tra". Provider coi
ngày chưa tra là `None` (chưa kiểm tra), không phải 0.

Chạy lại được: ngày đã tra được bỏ qua, dữ liệu cũ được giữ.

⚠ Script chưa được chạy thử trong repo này (cần MAP_KEY). Nếu API trả lỗi, script dừng
và in nguyên văn thông báo — ngày lỗi KHÔNG được ghi vào danh sách đã tra.
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import requests
import yaml

import _bootstrap  # noqa: F401
from config import DATA_DIR, get_settings

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
HANOI_LAT, HANOI_LON = 21.03, 105.85


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải điểm cháy FIRMS cho các ngày episode")
    parser.add_argument("--episodes", default=str(DATA_DIR / "eval" / "episodes.yaml"))
    parser.add_argument("--out", default=str(DATA_DIR / "raw" / "firms_hanoi.csv"))
    parser.add_argument(
        "--source",
        default="VIIRS_SNPP_SP",
        help="VIIRS_SNPP_SP (lưu trữ, dùng cho ngày cũ) | MODIS_SP | VIIRS_NOAA20_SP",
    )
    parser.add_argument(
        "--half-width-deg",
        type=float,
        default=3.2,
        help="Nửa bề rộng hộp quanh Hà Nội, độ. 3.2° ≈ 330 km, phủ bán kính 300 km",
    )
    args = parser.parse_args()

    key = get_settings().firms_map_key
    if not key:
        print(
            "✗ Chưa có FIRMS_MAP_KEY trong .env. Đăng ký: https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )
        return 2

    needed = _needed_dates(Path(args.episodes))
    out = Path(args.out)
    dates_file = out.with_name(out.name + ".dates")
    rows, queried = _load_existing(out, dates_file)
    todo = sorted(needed - queried)
    print(f"Cần {len(needed)} ngày, đã có {len(needed & queried)}, còn tra {len(todo)}.")

    w = HANOI_LON - args.half_width_deg
    e = HANOI_LON + args.half_width_deg
    s = HANOI_LAT - args.half_width_deg
    n = HANOI_LAT + args.half_width_deg
    area = f"{w:.2f},{s:.2f},{e:.2f},{n:.2f}"

    for day in todo:
        url = f"{API}/{key}/{args.source}/{area}/1/{day.isoformat()}"
        response = requests.get(url, timeout=90)
        text = response.text.strip()
        first_line = text.splitlines()[0] if text else ""
        if response.status_code != 200 or "latitude" not in first_line:
            print(f"✗ {day}: HTTP {response.status_code} — {first_line[:200]}")
            print("  Dừng lại. Các ngày đã tra thành công vẫn được lưu.")
            break
        new_rows = list(csv.DictReader(io.StringIO(text)))
        for row in new_rows:
            row["source"] = args.source
        rows.extend(new_rows)
        queried.add(day)
        print(f"  {day}: {len(new_rows)} điểm cháy")
        time.sleep(0.5)

    _save(out, dates_file, rows, queried)
    print(f"\nĐã lưu {len(rows)} điểm cháy → {out}")
    print(f"Ngày đã tra: {len(queried)} → {dates_file}")
    return 0


def _needed_dates(path: Path) -> set[Date]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("episodes", []) if isinstance(raw, dict) else raw
    needed: set[Date] = set()
    for item in items or []:
        value = item.get("date") if isinstance(item, dict) else None
        if value is None:
            continue
        day = value if isinstance(value, Date) else Date.fromisoformat(str(value))
        needed |= {day, day - timedelta(days=1)}
    return needed


def _load_existing(out: Path, dates_file: Path) -> tuple[list[dict], set[Date]]:
    rows: list[dict] = []
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    queried: set[Date] = set()
    if dates_file.exists():
        queried = {
            Date.fromisoformat(line.strip())
            for line in dates_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return rows, queried


def _save(out: Path, dates_file: Path, rows: list[dict], queried: set[Date]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = ["latitude", "longitude", "acq_date"]
    for row in rows:
        fields += [k for k in row if k not in fields]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dates_file.write_text(
        "\n".join(d.isoformat() for d in sorted(queried)) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
