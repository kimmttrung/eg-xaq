# 02 — Hợp đồng dữ liệu với lab (Data & Model Contract)

> **Đây là tài liệu quan trọng nhất ở giai đoạn hiện tại.** Toàn bộ code đang chạy bằng mock.
> File này định nghĩa chính xác cái "phích cắm" mà data/model của lab phải khớp vào.
>
> Trạng thái: ❌ **CHƯA CÓ DỮ LIỆU** — chờ chị mentor.
> Danh sách câu hỏi rút gọn để hỏi trực tiếp: `docs/08-questions-for-mentor.md`.

---

## 1. Những gì đã biết về hệ thống của lab

Từ mô tả ban đầu:

| Hạng mục | Đã biết |
|---|---|
| Thuật toán | XGBoost |
| Lưu trữ | Checkpoint `.pkl` |
| Số mô hình | **10 mô hình cho 10 step**, dự báo t → t+9 |
| Nguồn khí tượng | **GFS** (không phải ERA5) |
| Đặc trưng | Có biến tích lũy (accum): mean / max trên cửa sổ 2 ngày và 3 ngày |
| Cách suy luận | Đưa **bản đồ đặc trưng đầu vào** → mô hình dự báo **từng pixel** → bản đồ PM2.5 |
| Hiển thị | Web PopGIS, bản đồ AQI quy đổi từ PM2.5 |

Ba hệ quả quan trọng cho thiết kế:

1. **Mô hình là per-pixel, không phải per-station.** Nghĩa là với bất kỳ (lat, lon) nào trong
   miền dự báo, ta lấy được vector đặc trưng và chạy SHAP. Tốt cho đề tài: giải thích được ở
   mọi điểm, không chỉ nơi có trạm.
2. **10 model = 10 lời giải thích khác nhau.** SHAP cho t+0 và t+9 sẽ khác nhau về cấu trúc
   (t+0 phụ thuộc nhiều vào persistence, t+9 phụ thuộc nhiều vào khí tượng dự báo). Đây là một
   phân tích thú vị cho khóa luận: *độ giải thích suy giảm theo horizon như thế nào?*
3. **GFS chứ không phải ERA5** → ngưỡng rule phải hiệu chỉnh theo phân phối GFS. GFS là dữ liệu
   *dự báo*, ERA5 là *tái phân tích*; PBLH của hai nguồn có bias khác nhau. Xem §6.

---

## 2. Cần xin gì — Model artifacts

| # | Hạng mục | Vì sao cần | Bắt buộc |
|---|---|---|:---:|
| M1 | 10 file checkpoint + quy ước đặt tên (`model_step0.pkl` … ?) | Chọn đúng model cho step đang hỏi | ✅ |
| M2 | **Phiên bản** `python`, `xgboost`, `scikit-learn`, `numpy` lúc pickle | `pickle` rất dễ vỡ giữa các version. Nếu lệch → không load được | ✅ |
| M3 | Object bên trong pickle là gì: `XGBRegressor` (sklearn API) hay `xgboost.Booster` hay `Pipeline`? | Quyết định cách gọi `TreeExplainer` | ✅ |
| M4 | **Danh sách tên đặc trưng theo đúng thứ tự**, cho từng step | Sai thứ tự cột → SHAP ra kết quả vô nghĩa mà **không báo lỗi** | ✅ |
| M5 | Có scaler / normalizer không? Nếu có, artifact của nó | SHAP phải diễn giải trên **thang gốc** để nói được "PBLH 320 m" | ✅ |
| M6 | Target có biến đổi không (log1p, sqrt, chuẩn hóa)? | Phải nghịch đảo để quy về µg/m³ | ✅ |
| M7 | Giá trị `missing` XGBoost dùng (NaN hay -999?) | Pixel biển/nodata sẽ sai nếu hiểu nhầm | ✅ |
| M8 | Mẫu dữ liệu huấn luyện (~5.000–20.000 dòng) | (a) background cho SHAP interventional, (b) tính **anomaly** — "PBLH bình thường là bao nhiêu" | ✅ |
| M9 | Chỉ số đánh giá của mô hình (RMSE/MAE/R² theo step) | Phải báo cáo trong khóa luận: "hệ thống giải thích một mô hình có chất lượng X" | ✅ |
| M10 | Khoảng thời gian huấn luyện + tập test | Chọn episode đánh giá **ngoài** tập train, tránh leakage | ✅ |
| M11 | Có xin được luôn `.json`/`.ubj` (`booster.save_model()`) không? | Định dạng ổn định giữa version, tránh rủi ro M2 | ⭐ nên |

> ⚠️ **M4 là bẫy chết người.** XGBoost sklearn API không kiểm tra tên cột nếu ta đưa vào
> `numpy.ndarray`. Đưa sai thứ tự → vẫn chạy, vẫn ra số, SHAP vẫn đẹp, và **toàn bộ khóa luận
> sai**. Bắt buộc phải load `feature_names_in_` / `booster.feature_names` từ checkpoint và
> assert khớp, không hardcode. (Xem `CLAUDE.md` §8 bẫy #4.)

---

## 3. Cần xin gì — Feature engineering

| # | Hạng mục | Vì sao cần |
|---|---|---|
| F1 | **Code/notebook sinh đặc trưng từ GFS** | Để tái tạo được vector đặc trưng tại một điểm bất kỳ |
| F2 | Danh sách biến GFS thô: tên, **đơn vị**, mực (level) | Đổi đơn vị sai → rule sai (Kelvin vs °C, m vs mm) |
| F3 | Định nghĩa chính xác biến accum: cửa sổ 2/3 ngày là **lùi về quá khứ** hay bao gồm tương lai? Neo vào thời điểm nào? | Ảnh hưởng trực tiếp tới cách diễn giải: "gió yếu kéo dài 3 ngày" |
| F4 | Có đặc trưng tĩnh theo pixel không (độ cao, land use, mật độ dân, khoảng cách đường)? | Nếu có → **rất quan trọng**: đó là kênh mà mô hình học "pixel này vốn bẩn". Là ứng viên số 1 cho tương quan giả |
| F5 | Có đặc trưng thời gian không (day-of-year, tháng, sin/cos mùa)? | Cùng lý do F4 — mô hình có thể học "tháng 1 thì bẩn" thay vì cơ chế |
| F6 | Có dùng PM2.5 lag / persistence làm input không? | Nếu có, SHAP ở t+0 sẽ bị persistence chiếm hết → phải xử lý riêng khi diễn giải |
| F7 | Múi giờ: UTC hay giờ VN? Giờ khởi tạo GFS (00/06/12/18Z)? | Lệch 7 tiếng làm sai toàn bộ episode đánh giá |
| F8 | Step là **ngày** hay **giờ**? t+9 = 9 ngày hay 9 giờ? | Quyết định độ phân giải thời gian của toàn hệ thống |

---

## 4. Cần xin gì — Raster / lưới không gian

| # | Hạng mục | Vì sao cần |
|---|---|---|
| G1 | CRS (EPSG:4326?), độ phân giải, bounding box của miền dự báo | Ánh xạ (lat, lon) → chỉ số pixel |
| G2 | Định dạng bản đồ đặc trưng đầu vào: GeoTIFF nhiều band? NetCDF? `.npy`? | Quyết định thư viện đọc |
| G3 | Quy ước band ↔ đặc trưng | Cùng rủi ro như M4 |
| G4 | Giá trị nodata + mask (biển, ngoài miền) | Tránh giải thích một pixel rác |
| G5 | Lưu trữ ở đâu, truy cập thế nào (ổ mạng? S3? chỉ có trên server PopGIS?) | Ảnh hưởng kiến trúc: đọc file trực tiếp hay gọi API |

---

## 5. Cần xin gì — Web PopGIS & AQI

| # | Hạng mục | Vì sao cần |
|---|---|---|
| W1 | Có API lấy giá trị dự báo tại một điểm không? | Nếu có → có thể **không cần chạy lại mô hình**, chỉ cần SHAP. Đơn giản hóa rất nhiều |
| W2 | Công thức quy đổi PM2.5 → AQI đang dùng: **US EPA** hay **QCVN/VN AQI (QĐ 1459/QĐ-TCMT)**? | Hai thang khác nhau đáng kể. Đầu ra của hệ thống phải khớp với web, nếu không người dùng thấy mâu thuẫn |
| W3 | Có log truy vấn người dùng không? | Nguồn tốt để lấy câu hỏi thật cho bộ đánh giá |

---

## 6. Rủi ro đã nhận diện & cách xử lý

### R1. GFS ≠ ERA5 → ngưỡng rule lệch
Rule trong `knowledge/rules.yaml` đang đặt ngưỡng theo tài liệu quốc tế (chủ yếu dựa ERA5/quan
trắc). GFS có bias PBLH riêng.
**Xử lý**: sau khi có M8 (mẫu dữ liệu huấn luyện), chạy `scripts/calibrate_thresholds.py` để
đặt ngưỡng theo **phân vị của chính phân phối GFS** (ví dụ PBLH thấp = phân vị 10 của mùa đông
Hà Nội) thay vì con số tuyệt đối. Ngưỡng phân vị vừa đúng về mặt thống kê vừa dễ bảo vệ trước
hội đồng.

### R2. Mô hình per-pixel có thể học mẫu không gian thay vì cơ chế
Nếu F4/F5 cho biết có đặc trưng tĩnh hoặc thời gian, khả năng cao SHAP sẽ bị các biến đó chiếm
top. **Đây không phải lỗi cần giấu — đây là kết quả nghiên cứu.** Consistency check sẽ báo
`MODEL_ONLY` cho các biến đó, và khóa luận báo cáo: *"X% attribution của mô hình đến từ các biến
không mang cơ chế vật lý"*. Đó là một phát hiện có giá trị.

### R3. Pickle không load được do lệch version
**Xử lý**: xin M11 (`.json`/`.ubj`). Nếu không được, dựng venv riêng khớp version ở M2 chỉ để
chạy `xai/lab_xgb.py`, và cache kết quả SHAP ra file — phần còn lại của hệ thống không cần
xgboost.

### R4. Không được phép chạy lại mô hình trên máy cá nhân
**Xử lý**: dùng W1 (API PopGIS) cho giá trị dự báo, và xin lab chạy hộ SHAP theo lô cho ~50
episode đánh giá, xuất ra CSV theo schema ở §7.

---

## 7. Schema mà code đang chờ

Đây là "phích cắm". Bất kể lab giao gì, tầng provider phải chuyển đổi về đúng dạng này.

### 7.1. Observation — điều kiện tại (lat, lon, date, step)

```python
Observation(
    lat=21.03, lon=105.85, date=date(2024, 1, 15), step=0,
    pm25_pred_ugm3=95.2,        # dự báo của mô hình lab tại pixel này
    blh_m=320.0,                # planetary boundary layer height
    wind_speed_ms=0.8,          # tốc độ gió 10 m
    wind_dir_deg=45.0,          # hướng gió THỔI TỚI TỪ (meteorological convention)
    t2m_c=16.4,
    rh_pct=78.0,
    precip_mm=0.0,
    mslp_hpa=1024.0,
    t850_c=18.1,                # để phát hiện nghịch nhiệt; None nếu không có
    # cửa sổ tích lũy — khớp với biến accum của lab
    wind_speed_mean_2d_ms=1.1,
    wind_speed_mean_3d_ms=1.3,
    blh_min_2d_m=290.0,
    precip_sum_3d_mm=0.0,
    # nguồn ngoài
    upwind_fire_count=0,
    upwind_fire_distance_km=None,
    provenance="mock",          # 'mock' | 'lab_raster' | 'popgis_api'
)
```

Trường thiếu → `None`. Rule phụ thuộc trường `None` sẽ **không kích hoạt** và ghi vào
`bundle.missing` để narrator nói rõ hạn chế. Không bao giờ điền giá trị mặc định thay cho dữ liệu
thiếu.

### 7.2. Attribution — SHAP cho một điểm, một step

```python
Attribution(
    step=0,
    model_id="model_step0",
    base_value=42.1,            # expected_value của explainer, µg/m³
    prediction=95.2,
    contributions=[
        FeatureContribution(feature="blh_min_2d", value=290.0, shap=+18.4),
        FeatureContribution(feature="wind_speed_mean_2d", value=1.1, shap=+12.7),
        FeatureContribution(feature="precip_sum_3d", value=0.0, shap=+4.1),
        FeatureContribution(feature="rh_mean", value=78.0, shap=-2.0),
        ...
    ],
    provenance="mock",
)
```

Quy ước dấu: `shap > 0` nghĩa là đặc trưng đó **đẩy dự báo PM2.5 lên**. Đây là quy ước mặc định
của `shap` cho hồi quy — nhưng phải **kiểm chứng lại** với model thật vì nếu target là log-PM2.5
thì đơn vị SHAP là log, không phải µg/m³.

### 7.3. Ánh xạ tên đặc trưng ↔ biến cơ chế

Tên đặc trưng của lab (ví dụ `blh_min_2d`) khác tên biến trong `knowledge/mechanisms.yaml`
(ví dụ `blh`). Bảng ánh xạ nằm ở `knowledge/feature_map.yaml`, dạng:

```yaml
blh:        [blh, blh_min, blh_mean, blh_min_2d, blh_min_3d, pblh, hpbl]
wind_speed: [wind_speed, ws10, wind_speed_mean_2d, wind_speed_mean_3d]
precip:     [tp, precip, rain_sum, precip_sum_2d, precip_sum_3d]
```

Khi có M4 (danh sách đặc trưng thật), **việc đầu tiên** là cập nhật file này. Không cập nhật →
`shap_agreement` luôn bằng 0 và consistency check vô dụng.

---

## 8. Kế hoạch chuyển từ mock sang lab

```
1. Nhận artifacts        → điền §2–§5 vào docs này, đánh dấu ✅
2. Cập nhật feature_map.yaml theo M4
3. Viết src/egxaq/data/lab_raster.py   (implement ObservationProvider)
4. Viết src/egxaq/xai/lab_xgb.py       (implement AttributionProvider)
5. Chạy scripts/calibrate_thresholds.py trên M8 → cập nhật rules.yaml
6. Chạy lại toàn bộ test (test lõi không được đổi kết quả)
7. Đổi .env: EGXAQ_DATA_BACKEND=lab, EGXAQ_XAI_BACKEND=lab
8. Chạy tests/test_contract.py để xác thực provider thật khớp schema
```

Bước 6 là điểm kiểm soát: nếu test lõi vỡ khi cắm data thật, nghĩa là lõi đã lỡ phụ thuộc vào
mock ở đâu đó → phải sửa ngay, không được nới test.
