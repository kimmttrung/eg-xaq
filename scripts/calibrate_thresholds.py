"""Hiệu chỉnh ngưỡng rule theo phân phối dữ liệu THẬT của lab.

Vì sao cần bước này: ngưỡng trong `knowledge/rules.yaml` hiện là ngưỡng VĂN LIỆU,
chủ yếu rút từ nghiên cứu dùng ERA5 hoặc quan trắc ở nơi khác. Mô hình của lab
dùng GFS — nguồn dự báo, có bias riêng và khí hậu Hà Nội cũng khác. Một ngưỡng
PBLH 500 m có thể quá chặt hoặc quá lỏng với phân phối thực tế.

Ngưỡng theo PHÂN VỊ vừa đúng về thống kê vừa dễ bảo vệ trước hội đồng: "PBLH thấp"
được định nghĩa là "thuộc 15% thấp nhất của mùa đông Hà Nội theo dữ liệu GFS
2020–2024", chứ không phải một con số mượn từ bài báo về Bắc Kinh.

    python scripts/calibrate_thresholds.py --csv data/raw/lab_training_sample.csv
    python scripts/calibrate_thresholds.py --csv ... --date-col forecast_date

Rule nào có `calibration: percentile_*_winter` sẽ tự lọc theo tháng 11–3 dựa vào
cột ngày; các rule còn lại dùng toàn bộ dữ liệu.

Script CHỈ IN ĐỀ XUẤT, không tự sửa YAML — thay đổi tri thức phải do người quyết
định và phải giải thích được trong khóa luận.
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from kb import get_knowledge_base

#: `calibration` trong rules.yaml → (phân vị, có lọc theo mùa đông không)
CALIBRATION_SPECS: dict[str, tuple[float, bool]] = {
    "percentile_10": (10.0, False),
    "percentile_15_winter": (15.0, True),
    "percentile_20": (20.0, False),
    "percentile_80": (80.0, False),
    "percentile_80_winter": (80.0, True),
    "percentile_85": (85.0, False),
}

WINTER_MONTHS = {11, 12, 1, 2, 3}


def main() -> int:
    parser = argparse.ArgumentParser(description="Đề xuất ngưỡng rule theo phân vị dữ liệu thật")
    parser.add_argument("--csv", required=True, help="Mẫu dữ liệu huấn luyện từ lab (M8)")
    parser.add_argument("--date-col", default="date", help="Cột ngày, để lọc theo mùa")
    parser.add_argument(
        "--saturation-percentile-offset",
        type=float,
        default=5.0,
        help="Điểm bão hòa cách ngưỡng bao nhiêu phân vị (mặc định 5)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        winter = df[df[args.date_col].dt.month.isin(WINTER_MONTHS)]
    else:
        print(f"⚠ Không có cột '{args.date_col}' — bỏ qua lọc mùa.")
        winter = df

    kb = get_knowledge_base()
    print(f"Dữ liệu: {len(df)} dòng ({len(winter)} dòng mùa đông)\n")
    print(f"{'Rule':<5} {'Biến':<24} {'Hiện tại':>20}   {'Đề xuất':>20}")
    print("-" * 76)

    changes = 0
    for rule in kb.rules.values():
        spec = CALIBRATION_SPECS.get(rule.calibration)
        if spec is None:
            continue

        percentile, winter_only = spec
        frame = winter if winter_only else df
        column = _find_column(frame, rule.variable)
        if column is None:
            print(f"{rule.id:<5} {rule.variable:<24} {'—':>20}   không có cột tương ứng")
            continue

        series = frame[column].dropna()
        if len(series) < 100:
            print(f"{rule.id:<5} {rule.variable:<24} {'—':>20}   quá ít mẫu ({len(series)})")
            continue

        offset = args.saturation_percentile_offset
        if rule.direction == "below":
            threshold = series.quantile(percentile / 100.0)
            saturation = series.quantile(max(0.5, percentile - offset) / 100.0)
        else:
            threshold = series.quantile(percentile / 100.0)
            saturation = series.quantile(min(99.5, percentile + offset) / 100.0)

        current = f"{rule.threshold:g} → {rule.saturation:g}"
        proposed = f"{threshold:.4g} → {saturation:.4g}"
        flag = "  ← khác đáng kể" if _differs(rule.threshold, threshold) else ""
        print(f"{rule.id:<5} {rule.variable:<24} {current:>20}   {proposed:>20}{flag}")
        changes += bool(flag)

    print("\n" + "-" * 76)
    print(f"{changes} ngưỡng lệch đáng kể so với dữ liệu thật.")
    print(
        "Cập nhật knowledge/rules.yaml THỦ CÔNG, ghi lý do vào trường `source`,\n"
        "rồi đặt `calibrated: true` và sửa test test_rules_yaml_marked_uncalibrated."
    )
    return 0


def _find_column(df: pd.DataFrame, variable: str) -> str | None:
    """Khớp tên biến của rule với tên cột trong CSV của lab.

    Thử khớp chính xác trước, rồi bỏ hậu tố đơn vị (`_m`, `_ms`, `_pct`, `_hpa`),
    rồi khớp theo tiền tố. Cột nào không khớp sẽ được báo để xử lý tay — đoán mò
    ở đây nguy hiểm hơn là bỏ sót.
    """
    if variable in df.columns:
        return variable

    stem = variable
    for suffix in ("_m2s", "_c_per_km", "_hpa", "_pct", "_ms", "_mm", "_m", "_c"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if stem in df.columns:
        return stem
    matches = [c for c in df.columns if c.lower().startswith(stem.lower())]
    return matches[0] if len(matches) == 1 else None


def _differs(current: float, proposed: float, tolerance: float = 0.25) -> bool:
    if current == 0:
        return proposed != 0
    return abs(proposed - current) / abs(current) > tolerance


if __name__ == "__main__":
    raise SystemExit(main())
