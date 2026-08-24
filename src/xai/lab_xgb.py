"""AttributionProvider chạy SHAP trên 10 checkpoint XGBoost của lab.

⚠ CHƯA CÀI ĐẶT — chờ checkpoint từ chị mentor.

Khung này ghi lại toàn bộ quyết định kỹ thuật cần lưu ý, để lúc có model chỉ phải
điền phần đọc file. Xem docs/02-data-contract.md §2.

Cài đặt xong thì đổi `.env`:  EGXAQ_XAI_BACKEND=lab
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from config import get_settings
from schemas import Attribution, Observation


class LabXGBAttributionProvider:
    """TreeSHAP trên checkpoint tương ứng với bước dự báo đang hỏi.

    ---------------------------------------------------------------------------
    QUYẾT ĐỊNH KỸ THUẬT ĐÃ CHỐT (đọc trước khi code)
    ---------------------------------------------------------------------------

    **Chọn đúng model theo step.** Lab có 10 checkpoint cho t → t+9. Giải thích
    dự báo t+3 bằng model của t+0 là sai hoàn toàn. Cache model đã load trong
    `self._models` vì mỗi lần unpickle XGBoost tốn cả trăm ms.

    **ĐỌC feature_names TỪ CHECKPOINT, không hardcode.** Đây là bẫy nguy hiểm
    nhất của cả dự án: XGBoost nhận `numpy.ndarray` mà không kiểm tra tên cột.
    Đưa sai thứ tự → vẫn chạy, vẫn ra số đẹp, SHAP vẫn vẽ được, và toàn bộ kết
    luận của khóa luận sai mà không có dấu hiệu nào. Bắt buộc:

        names = model.get_booster().feature_names   # hoặc model.feature_names_in_
        assert list(feature_frame.columns) == list(names)

    **TreeExplainer, không KernelExplainer.** TreeSHAP là thuật toán chính xác
    cho mô hình cây, chạy trong thời gian đa thức. KernelExplainer chỉ là xấp xỉ
    dựa trên lấy mẫu, chậm hơn hàng nghìn lần và có phương sai — không có lý do
    dùng nó ở đây.

        explainer = shap.TreeExplainer(model)       # tree_path_dependent
        values = explainer.shap_values(X)

    Nếu muốn SHAP "interventional" (diễn giải nhân quả chặt hơn) thì cần
    `data=background` với mẫu dữ liệu huấn luyện (M8). Chậm hơn nhưng đúng hơn về
    lý thuyết. Nên thử cả hai và báo cáo trong khóa luận.

    **Đơn vị của giá trị SHAP.** Nếu target được huấn luyện trên log1p(PM2.5)
    (M6), giá trị SHAP nằm trên thang log — KHÔNG phải µg/m³. Khi đó phải ghi rõ
    vào `Attribution.provenance` để narrator không nói sai đơn vị. Không được
    "quy đổi ngược" từng giá trị SHAP: phép biến đổi phi tuyến làm tổng các phần
    không còn bằng hiệu của tổng.

    **Giải thích một điểm, không giải thích cả bản đồ.** Mô hình dự báo per-pixel
    trên toàn raster; chạy SHAP cho mọi pixel là vô nghĩa về chi phí lẫn mục đích.
    Chỉ giải thích tại điểm người dùng hỏi.
    """

    name = "lab_xgb"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        settings = get_settings()
        raw = model_dir or settings.lab_model_dir
        if not raw:
            raise NotImplementedError(
                "Chưa cấu hình EGXAQ_LAB_MODEL_DIR và chưa có checkpoint từ lab. "
                "Xem docs/02-data-contract.md §2."
            )
        self.model_dir = Path(raw)
        self._models: dict[int, object] = {}
        self._explainers: dict[int, object] = {}

    def get(
        self,
        lat: float,
        lon: float,
        date: Date,
        step: int = 0,
        observation: Observation | None = None,
    ) -> Attribution:
        raise NotImplementedError(
            "LabXGBAttributionProvider chưa được cài đặt.\n"
            "Cần từ lab (docs/02-data-contract.md §2):\n"
            "  M1  10 checkpoint + quy ước đặt tên\n"
            "  M2  phiên bản python/xgboost/sklearn lúc pickle\n"
            "  M3  object bên trong pickle là gì\n"
            "  M4  danh sách feature theo ĐÚNG thứ tự, cho từng step\n"
            "  M5  scaler (nếu có)\n"
            "  M6  target có biến đổi không (log1p?)\n"
            "  M8  mẫu dữ liệu huấn luyện làm background cho SHAP\n"
            "Trong lúc chờ, dùng EGXAQ_XAI_BACKEND=mock."
        )

    # ------------------------------------------------------------------ khung
    def _load_model(self, step: int):
        """Load và cache checkpoint của một bước dự báo.

        Nếu pickle không mở được do lệch phiên bản (rủi ro R3 trong data contract),
        xin lab xuất thêm `booster.save_model('model.json')` — định dạng đó ổn định
        giữa các phiên bản, còn pickle thì không.
        """
        raise NotImplementedError

    def _feature_frame(self, observation: Observation, step: int):
        """Dựng DataFrame một dòng theo ĐÚNG thứ tự cột mà model mong đợi.

        Nguồn giá trị phải là chính raster đặc trưng của lab, KHÔNG phải các trường
        đã chuẩn hóa trong `Observation` — vì `Observation` có thể đã đổi đơn vị và
        làm tròn, còn model được huấn luyện trên số thô.
        """
        raise NotImplementedError
