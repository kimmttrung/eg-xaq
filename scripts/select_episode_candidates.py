"""Chọn ngày ứng viên cho bộ đánh giá từ PM2.5 quan trắc, sinh phiếu gán nhãn.

    python scripts/select_episode_candidates.py --csv ../era5/dataset_train.csv
    python scripts/select_episode_candidates.py --csv ... --per-group 12 --min-gap 7

Sinh ra data/eval/episodes.yaml với mọi mục ở `status: todo`. Bạn đọc nguồn, điền
nhãn, rồi đổi status thành `labeled` (hoặc `rejected` nếu không tìm được nguồn).

Vì sao chọn theo PM2.5 chứ không theo khí tượng: xem docstring đầu src/evaluation.py.
CSV chỉ cần hai cột `date` và `pm25`; các cột khí tượng bị bỏ qua có chủ đích.

Script KHÔNG ghi đè file nhãn đã có — nhãn là công sức gán tay, mất là mất hẳn.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date as Date
from pathlib import Path

import _bootstrap  # noqa: F401
from config import DATA_DIR
from evaluation import (
    DailyRecord,
    render_worksheet,
    sample_for_second_annotator,
    select_candidates,
)
from kb import get_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Chọn ngày ứng viên theo PM2.5 quan trắc")
    parser.add_argument(
        "--csv", required=True, help="CSV có cột date, pm25 (µg/m³, trung bình ngày)"
    )
    parser.add_argument("--per-group", type=int, default=15)
    parser.add_argument(
        "--min-gap", type=int, default=5, help="Khoảng cách tối thiểu giữa hai ngày"
    )
    parser.add_argument("--out", default=None, help="Mặc định data/eval/episodes.yaml")
    parser.add_argument("--force", action="store_true", help="Cho phép ghi đè (MẤT nhãn đã gán)")
    parser.add_argument(
        "--second-annotator",
        type=int,
        default=15,
        help="Số ngày cho người gán nhãn thứ hai (0 = không sinh phiếu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed chọn tập con cho người thứ hai")
    args = parser.parse_args()

    out = Path(args.out) if args.out else DATA_DIR / "eval" / "episodes.yaml"
    if out.exists() and not args.force:
        print(f"✗ {out} đã tồn tại — có thể đã chứa nhãn gán tay. Dùng --out khác, hoặc --force.")
        return 1

    records = _read_records(Path(args.csv))
    if not records:
        print("✗ Không đọc được dòng nào có PM2.5.")
        return 1

    candidates = select_candidates(records, per_group=args.per_group, min_gap_days=args.min_gap)
    first, last = min(r.date for r in records), max(r.date for r in records)
    note = f"{Path(args.csv).name}, {len(records)} ngày ({first} → {last})"

    out.parent.mkdir(parents=True, exist_ok=True)
    kb = get_knowledge_base()
    out.write_text(render_worksheet(candidates, kb, note), encoding="utf-8")

    second_path, subset = None, []
    if args.second_annotator > 0:
        second_path = out.with_name(out.stem + ".annotator2" + out.suffix)
        if second_path.exists() and not args.force:
            print(f"⚠ {second_path} đã tồn tại — không ghi đè phiếu người thứ hai.")
            second_path = None
        else:
            subset = sample_for_second_annotator(candidates, args.second_annotator, args.seed)
            second_path.write_text(
                render_worksheet(subset, kb, note, second_annotator=True), encoding="utf-8"
            )

    print(f"Dữ liệu: {note}")
    print(f"Đã chọn {len(candidates)} ngày → {out}\n")
    for group, count in Counter(c.group for c in candidates).items():
        values = [c.pm25_ugm3 for c in candidates if c.group == group]
        print(f"  {group:<22} {count:>3} ngày   PM2.5 {min(values):.0f}–{max(values):.0f} µg/m³")
    if second_path:
        print(f"\nPhiếu người thứ hai: {len(subset)} ngày → {second_path} (seed {args.seed})")
    print("\nBước tiếp theo: đọc data/eval/LABELING.md rồi gán nhãn từng ngày.")
    return 0


def _read_records(path: Path) -> list[DailyRecord]:
    if not path.exists():
        raise SystemExit(f"✗ Không tìm thấy {path}")
    records: list[DailyRecord] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("pm25") or "").strip()
            if not raw:
                continue  # ngày thiếu quan trắc: bỏ, KHÔNG điền giá trị
            records.append(DailyRecord(Date.fromisoformat(row["date"][:10]), float(raw)))
    return records


if __name__ == "__main__":
    raise SystemExit(main())
