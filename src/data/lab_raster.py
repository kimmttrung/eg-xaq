"""ObservationProvider đọc raster đặc trưng GFS của lab.

⚠ CHƯA CÀI ĐẶT — chờ dữ liệu từ chị mentor.

File này là KHUNG có chủ đích: nó ghi rõ từng chỗ cần điền và vì sao, để khi nhận
được artifacts thì việc nối vào là điền chỗ trống chứ không phải thiết kế lại.
Danh sách chính xác những gì cần xin: docs/02-data-contract.md §3–§5.

Cài đặt xong thì đổi `.env`:  EGXAQ_DATA_BACKEND=lab
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from config import get_settings
from schemas import Observation


class LabRasterObservationProvider:
    """Đọc giá trị đặc trưng tại một pixel từ bản đồ đầu vào của mô hình lab.

    ---------------------------------------------------------------------------
    CẦN ĐIỀN KHI CÓ DỮ LIỆU
    ---------------------------------------------------------------------------

    1. `_open_stack(date, step)` — mở raster đặc trưng cho ngày/bước đó.
       Cần biết: định dạng (GeoTIFF nhiều band? NetCDF?), quy ước đặt tên file,
       và **band nào ứng với đặc trưng nào** (G2, G3 trong data contract).

    2. `_pixel_index(lat, lon)` — (lat, lon) → (row, col).
       Cần biết: CRS, độ phân giải, bounding box (G1). Nếu dùng rasterio thì
       `dataset.index(lon, lat)` lo phần này, nhưng PHẢI kiểm tra CRS trước —
       raster ở UTM mà truyền vào lat/lon sẽ cho ra chỉ số sai mà không báo lỗi.

    3. `_to_observation(values)` — vector đặc trưng thô → `Observation`.
       Đây là chỗ ĐỔI ĐƠN VỊ. Ba bẫy đã biết (CLAUDE.md §8):
         - GFS trả nhiệt độ Kelvin, `Observation.t2m_c` cần Celsius.
         - Mưa tích lũy có thể là mét hoặc kg/m²; `precip_mm` cần milimét.
         - u10/v10 → dùng `geoutils.uv_to_speed_dir`, ĐỪNG tự tính lại.

    4. Giá trị nodata (G4) → để `None`, không để 0. Một pixel biển có PBLH = 0
       sẽ kích hoạt rule "lớp xáo trộn thấp" ở mức tối đa và sinh ra một lời
       giải thích hoàn toàn bịa.

    5. `blh_climatology_m` — cần chuỗi khí hậu để tính anomaly. Có thể tính từ
       mẫu dữ liệu huấn luyện (M8): trung bình PBLH theo tháng tại pixel đó.
       Không có thì để `None`, hệ thống vẫn chạy, chỉ mất phần so sánh "thấp hơn
       bình thường X%" trong câu trả lời.
    """

    name = "lab_raster"

    def __init__(self, raster_dir: str | Path | None = None) -> None:
        settings = get_settings()
        raw = raster_dir or settings.lab_raster_dir
        if not raw:
            raise NotImplementedError(
                "Chưa cấu hình EGXAQ_LAB_RASTER_DIR và chưa có dữ liệu raster từ lab. "
                "Xem docs/02-data-contract.md §4."
            )
        self.raster_dir = Path(raw)

    def get(self, lat: float, lon: float, date: Date, step: int = 0) -> Observation:
        raise NotImplementedError(
            "LabRasterObservationProvider chưa được cài đặt.\n"
            "Cần từ lab (docs/02-data-contract.md):\n"
            "  G1 CRS + độ phân giải + bounding box\n"
            "  G2 định dạng raster đặc trưng\n"
            "  G3 quy ước band ↔ đặc trưng\n"
            "  G4 giá trị nodata\n"
            "  F2 tên biến GFS + ĐƠN VỊ\n"
            "Trong lúc chờ, dùng EGXAQ_DATA_BACKEND=mock."
        )


class PopGISObservationProvider:
    """Lấy giá trị dự báo qua API của web PopGIS thay vì chạy lại mô hình.

    Nếu lab có sẵn endpoint truy vấn theo điểm (W1 trong data contract) thì đây là
    đường ngắn hơn hẳn: không cần raster, không cần load 10 checkpoint, không cần
    khớp phiên bản xgboost. Đánh đổi: chỉ lấy được GIÁ TRỊ DỰ BÁO, không lấy được
    vector đặc trưng — nên vẫn phải có nguồn khí tượng riêng cho rule engine, và
    SHAP vẫn cần checkpoint thật.
    """

    name = "popgis_api"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url

    def get(self, lat: float, lon: float, date: Date, step: int = 0) -> Observation:
        raise NotImplementedError(
            "Chưa có endpoint PopGIS. Cần hỏi mentor: W1 và W2 trong "
            "docs/02-data-contract.md §5."
        )
