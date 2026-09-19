"""Tải bổ sung ERA5: nhiệt độ mực 850 hPa + áp suất mực biển, gộp theo ngày tại Hà Nội.

    pip install cdsapi xarray netCDF4
    python scripts/download_era5_supplement.py --start 2021-12 --end 2024-12

Vì sao cần: ../era5/features_daily.csv không có hai biến này, nên nghịch nhiệt (R3, cần
t850) và cao áp (R11, cần áp suất) luôn "không kiểm tra được". Hai nhóm episode mùa đông
sẽ bị chấm thấp vì thiếu dữ liệu chứ không phải vì hệ thống sai.

Cần tài khoản Copernicus CDS và file ~/.cdsapirc — dùng chung với ../era5/download_era5.py.

Nhất quán với chuỗi đã có (../era5/build_features.py):
- cùng hộp tải [22, 105, 20, 107] và cùng cách chọn điểm lưới gần (21.03, 105.85) nhất
- cùng cách gom 24 giờ theo ngày UTC
- ĐỔI ĐƠN VỊ ngay tại đây (CLAUDE.md bẫy #1): nhiệt độ K → °C, áp suất Pa → hPa

Ghi data/raw/era5_supplement_daily.csv với cột:
    date, t850_mean_c, t850_00utc_c, mslp_mean_hpa
`t850_00utc_c` (07:00 sáng Hà Nội) được giữ lại để có thể tính lapse rate buổi sáng —
lúc nghịch nhiệt mạnh nhất — mà không phải tải lại.

⚠ Script chưa được chạy thử trong repo này (cần tài khoản CDS). File .nc tải về được
giữ trong data/raw/era5_supplement/ nên chạy lại chỉ gộp, không tải lại.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from config import DATA_DIR

AREA = [22.0, 105.0, 20.0, 107.0]  # [Bắc, Tây, Nam, Đông] — giống ../era5/download_era5.py
HANOI_LAT, HANOI_LON = 21.03, 105.85
TIME_DIM = "valid_time"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải t850 + áp suất mực biển ERA5")
    parser.add_argument("--start", default="2021-12", help="YYYY-MM")
    parser.add_argument("--end", default="2024-12", help="YYYY-MM")
    parser.add_argument("--raw-dir", default=str(DATA_DIR / "raw" / "era5_supplement"))
    parser.add_argument("--out", default=str(DATA_DIR / "raw" / "era5_supplement_daily.csv"))
    args = parser.parse_args()

    try:
        import cdsapi
        import xarray as xr
    except ImportError:
        print("✗ Cần: pip install cdsapi xarray netCDF4")
        return 2

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year, month in _months(args.start, args.end):
        tag = f"{year}{month:02d}"
        base = {
            "product_type": ["reanalysis"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        t850_file = raw_dir / f"t850_{tag}.nc"
        if not t850_file.exists():
            print(f"Tải t850 {tag} …")
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {**base, "variable": ["temperature"], "pressure_level": ["850"]},
                str(t850_file),
            )
        msl_file = raw_dir / f"msl_{tag}.nc"
        if not msl_file.exists():
            print(f"Tải áp suất {tag} …")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {**base, "variable": ["mean_sea_level_pressure"]},
                str(msl_file),
            )

    t850 = _hourly_point(xr, sorted(raw_dir.glob("t850_*.nc")), "t")
    msl = _hourly_point(xr, sorted(raw_dir.glob("msl_*.nc")), "msl")

    t850_c = t850 - 273.15  # K → °C
    msl_hpa = msl / 100.0  # Pa → hPa

    daily_t = t850_c.groupby(t850_c.index.date)
    morning = t850_c[t850_c.index.hour == 0]
    rows = {}
    for day, values in daily_t:
        rows.setdefault(day, {})["t850_mean_c"] = float(values.mean())
    for stamp, value in morning.items():
        rows.setdefault(stamp.date(), {})["t850_00utc_c"] = float(value)
    for day, values in msl_hpa.groupby(msl_hpa.index.date):
        rows.setdefault(day, {})["mslp_mean_hpa"] = float(values.mean())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["date", "t850_mean_c", "t850_00utc_c", "mslp_mean_hpa"]
        )
        writer.writeheader()
        for day in sorted(rows):
            writer.writerow({"date": day.isoformat(), **rows[day]})

    print(f"\nĐã ghi {len(rows)} ngày → {out}")
    print("Kiểm tra nhanh khoảng giá trị (Hà Nội): t850 thường 0–25 °C, áp suất 995–1035 hPa.")
    return 0


def _months(start: str, end: str):
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            year, month = year + 1, 1


def _hourly_point(xr, files, variable: str):
    """Ghép các file theo thời gian, lấy điểm lưới gần Hà Nội nhất, trả pandas Series."""
    if not files:
        raise SystemExit(f"✗ Không có file nào cho biến {variable}")
    parts = []
    for path in files:
        ds = xr.open_dataset(path)
        for extra in ("number", "expver"):
            if extra in ds.coords:
                ds = ds.drop_vars(extra)
        point = ds.sel(latitude=HANOI_LAT, longitude=HANOI_LON, method="nearest")
        if "pressure_level" in point.dims:
            point = point.squeeze("pressure_level", drop=True)
        parts.append(point[variable].to_series())
    series = parts[0] if len(parts) == 1 else __import__("pandas").concat(parts)
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series.index = __import__("pandas").to_datetime(series.index)
    return series


if __name__ == "__main__":
    raise SystemExit(main())
