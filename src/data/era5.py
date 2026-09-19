"""ObservationProvider đọc chuỗi ERA5 theo ngày tại Hà Nội.

Nguồn khí tượng THẬT trong lúc chờ GFS của lab — phương án dự phòng số 1 (docs/07 §3).
Dùng để chấm bộ episode đánh giá; KHÔNG phải nguồn của mô hình dự báo, nên không có
`pm25_pred_ugm3` và không có attribution đi kèm.

NGUỒN VÀ ĐƠN VỊ — đã đối chiếu với ../era5/build_features.py
============================================================
    features_daily.csv   điểm lưới gần (21.03, 105.85) nhất, gom 24 giờ theo ngày UTC
        blh_mean         m        trung bình ngày
        wind_speed_mean  m/s      trung bình của tốc độ giờ
        wind_dir_mean    độ       từ trung bình vector u10/v10, quy ước "thổi từ đâu tới"
        t2m_mean         °C       đã trừ 273.15
        rh_mean          %        Magnus từ t2m và d2m
        rain_sum         mm/ngày  tp × 1000 rồi cộng theo giờ

    era5_supplement_daily.csv  (scripts/download_era5_supplement.py)
        t850_mean_c      °C       nhiệt độ mực 850 hPa, trung bình ngày
        mslp_mean_hpa    hPa      áp suất mực biển, trung bình ngày

    firms_hanoi.csv + firms_hanoi.csv.dates  (scripts/download_firms.py)
    dataset_train.csv    cột pm25, µg/m³, OpenAQ — Hanoi Air Quality Monitoring Network

HẠN CHẾ PHẢI NÊU
================
1. Ngày gom theo UTC (07:00 → 07:00 giờ Hà Nội), không theo lịch địa phương.
2. Lapse rate tính từ TRUNG BÌNH NGÀY của t2m và t850. Trung bình ngày gồm cả buổi chiều
   đã nóng lên, nên nghịch nhiệt ban đêm/sáng sớm bị làm mờ → R3 thận trọng hơn thực tế.
   File bổ sung có sẵn `t850_00utc_c` (07:00 sáng) nếu cần đổi cách tính.
3. `blh_min_2d_m` / `blh_min_3d_m` là cực tiểu của TRUNG BÌNH NGÀY, không phải cực tiểu
   theo giờ — cực tiểu giờ là PBLH ban đêm (vài chục mét), khác hẳn nghĩa của rule.
4. Giá trị khí hậu nền (anomaly) tính trên toàn chuỗi 2017–2024, gồm cả ngày đang xét.

NGUYÊN TẮC THIẾU DỮ LIỆU (INV-3)
================================
- Thiếu một biến, hoặc thiếu một ngày trong cửa sổ 2–3 ngày → trường đó là `None`. Không
  tính trên phần còn lại của cửa sổ, không điền số.
- Chưa tải FIRMS cho ngày đó → `upwind_fire_count = None`, KHÔNG phải 0. "Chưa kiểm tra"
  khác "đã kiểm tra, không có điểm cháy" — vì vậy file FIRMS phải kèm danh sách ngày đã tra.
- Không có dòng nào cho ngày được hỏi, hỏi ngoài Hà Nội, hoặc hỏi step > 0 →
  `ObservationUnavailable`. Đó là lỗi cấu hình, không phải dữ liệu thiếu.
"""

from __future__ import annotations

import csv
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from geoutils import haversine_km, is_upwind
from schemas import Observation

from .base import ObservationUnavailable

HANOI_LAT, HANOI_LON = 21.03, 105.85

#: Chuỗi chỉ đại diện cho một điểm lưới 0.25° (~28 km). Hỏi xa hơn thế là hỏi nơi khác.
MAX_OFFSET_KM = 30.0

FEATURE_COLUMNS = (
    "blh_mean",
    "wind_speed_mean",
    "wind_dir_mean",
    "t2m_mean",
    "rh_mean",
    "rain_sum",
)
SUPPLEMENT_COLUMNS = ("t850_mean_c", "mslp_mean_hpa")


def _to_float(raw: str | None) -> float | None:
    text = (raw or "").strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def _read_daily(path: Path, columns: tuple[str, ...]) -> dict[Date, dict[str, float | None]]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("date", *columns) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: thiếu cột {missing}")
        return {
            Date.fromisoformat(row["date"][:10]): {c: _to_float(row.get(c)) for c in columns}
            for row in reader
        }


def _read_firms(path: Path) -> tuple[dict[Date, list[tuple[float, float]]], set[Date]]:
    """Điểm cháy theo ngày + tập ngày ĐÃ TRA. Thiếu file ngày đã tra → không dùng được."""
    dates_file = path.with_name(path.name + ".dates")
    if not dates_file.exists():
        raise ValueError(
            f"{dates_file} không tồn tại. Không có danh sách ngày đã tra thì không phân biệt "
            "được 'không có điểm cháy' với 'chưa tra'. Tải lại bằng scripts/download_firms.py."
        )
    queried = {
        Date.fromisoformat(line.strip())
        for line in dates_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    fires: dict[Date, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            day = Date.fromisoformat(row["acq_date"][:10])
            fires.setdefault(day, []).append((float(row["latitude"]), float(row["longitude"])))
    return fires, queried


class ERA5DailyObservationProvider:
    """Cài đặt `ObservationProvider` trên chuỗi ERA5 theo ngày."""

    name = "era5"

    def __init__(
        self,
        features_csv: Path | str,
        supplement_csv: Path | str | None = None,
        firms_csv: Path | str | None = None,
        pm25_csv: Path | str | None = None,
        upwind_sector_deg: float = 90.0,
        upwind_radius_km: float = 300.0,
    ) -> None:
        features_path = Path(features_csv)
        if not features_path.exists():
            raise FileNotFoundError(f"Không tìm thấy chuỗi ERA5: {features_path}")
        self.features = _read_daily(features_path, FEATURE_COLUMNS)
        self.sector_deg = upwind_sector_deg
        self.radius_km = upwind_radius_km

        #: Nguồn phụ được khai báo nhưng chưa có file. Được in ra lúc chấm điểm — biến
        #: tương ứng sẽ là None, và điều đó phải NHÌN THẤY được.
        self.missing_sources: list[str] = []
        self._sources = ["features"]

        self.supplement = self._optional(
            supplement_csv,
            lambda p: _read_daily(p, SUPPLEMENT_COLUMNS),
            "t850 + áp suất",
            "nghịch nhiệt (R3) và cao áp (R11) sẽ không kiểm tra được",
        )
        firms = self._optional(
            firms_csv,
            _read_firms,
            "FIRMS",
            "điểm cháy thượng nguồn (R6) sẽ không kiểm tra được",
        )
        self.fires, self.fire_dates = firms if firms else (None, None)
        pm25 = self._optional(pm25_csv, lambda p: _read_daily(p, ("pm25",)), "PM2.5 quan trắc", "")
        self.pm25 = pm25

        self._climatology = self._monthly_climatology()

    def _optional(self, raw, reader, label: str, consequence: str):
        if raw is None:
            return None
        path = Path(raw)
        if not path.exists():
            note = f"Chưa có file {label}: {path}"
            self.missing_sources.append(f"{note} — {consequence}" if consequence else note)
            return None
        self._sources.append(label)
        return reader(path)

    # ------------------------------------------------------------------ chính
    def get(self, lat: float, lon: float, date: Date, step: int = 0) -> Observation:
        if step != 0:
            raise ObservationUnavailable(
                "ERA5 là tái phân tích, không phải dự báo — chỉ hỗ trợ step = 0."
            )
        offset_km = haversine_km(lat, lon, HANOI_LAT, HANOI_LON)
        if offset_km > MAX_OFFSET_KM:
            raise ObservationUnavailable(
                f"Chuỗi ERA5 chỉ có điểm Hà Nội; ({lat}, {lon}) cách {offset_km:.0f} km."
            )
        today = self.features.get(date)
        if today is None:
            first, last = min(self.features), max(self.features)
            raise ObservationUnavailable(
                f"Không có ERA5 ngày {date} (chuỗi hiện có {first} → {last})."
            )

        wind_dir = today["wind_dir_mean"]
        wind_2d = self._window(date, 2, "wind_speed_mean")
        wind_3d = self._window(date, 3, "wind_speed_mean")
        blh_2d = self._window(date, 2, "blh_mean")
        blh_3d = self._window(date, 3, "blh_mean")
        rain_3d = self._window(date, 3, "rain_sum")
        extra = (self.supplement or {}).get(date, {})
        fire_count, fire_distance = self._upwind_fires(lat, lon, date, wind_dir)
        blh_clim, wind_clim = self._climatology.get(date.month, (None, None))

        return Observation(
            lat=lat,
            lon=lon,
            date=date,
            step=step,
            pm25_obs_ugm3=((self.pm25 or {}).get(date) or {}).get("pm25"),
            blh_m=today["blh_mean"],
            wind_speed_ms=today["wind_speed_mean"],
            wind_dir_deg=wind_dir,
            t2m_c=today["t2m_mean"],
            t850_c=extra.get("t850_mean_c"),
            rh_pct=today["rh_mean"],
            precip_mm=today["rain_sum"],
            mslp_hpa=extra.get("mslp_mean_hpa"),
            wind_speed_mean_2d_ms=sum(wind_2d) / 2 if wind_2d else None,
            wind_speed_mean_3d_ms=sum(wind_3d) / 3 if wind_3d else None,
            blh_min_2d_m=min(blh_2d) if blh_2d else None,
            blh_min_3d_m=min(blh_3d) if blh_3d else None,
            precip_sum_3d_mm=sum(rain_3d) if rain_3d else None,
            upwind_fire_count=fire_count,
            upwind_fire_distance_km=fire_distance,
            blh_climatology_m=blh_clim,
            wind_speed_climatology_ms=wind_clim,
            provenance="era5:" + "+".join(self._sources),
        )

    # ------------------------------------------------------------------ phụ trợ
    def _window(self, date: Date, days: int, column: str) -> list[float] | None:
        """Giá trị `days` ngày tính đến hết `date`. Thiếu bất kỳ ngày nào → None."""
        values: list[float] = []
        for back in range(days):
            row = self.features.get(date - timedelta(days=back))
            value = row.get(column) if row else None
            if value is None:
                return None
            values.append(value)
        return values

    def _upwind_fires(
        self, lat: float, lon: float, date: Date, wind_dir: float | None
    ) -> tuple[int | None, float | None]:
        """Điểm cháy trong ngày và ngày hôm trước, nằm trong quạt thượng nguồn gió.

        Tính cả hôm trước vì khói cần thời gian di chuyển: 300 km với gió 3 m/s mất
        khoảng một ngày.
        """
        if self.fires is None or wind_dir is None:
            return None, None
        days = (date, date - timedelta(days=1))
        if any(day not in self.fire_dates for day in days):
            return None, None

        distances = []
        for day in days:
            for fire_lat, fire_lon in self.fires.get(day, []):
                upwind, distance = is_upwind(
                    lat, lon, fire_lat, fire_lon, wind_dir, self.sector_deg, self.radius_km
                )
                if upwind:
                    distances.append(distance)
        return len(distances), (min(distances) if distances else None)

    def _monthly_climatology(self) -> dict[int, tuple[float | None, float | None]]:
        sums: dict[int, list[list[float]]] = {}
        for day, row in self.features.items():
            bucket = sums.setdefault(day.month, [[], []])
            if row["blh_mean"] is not None:
                bucket[0].append(row["blh_mean"])
            if row["wind_speed_mean"] is not None:
                bucket[1].append(row["wind_speed_mean"])
        return {
            month: (
                sum(blh) / len(blh) if blh else None,
                sum(wind) / len(wind) if wind else None,
            )
            for month, (blh, wind) in sums.items()
        }
