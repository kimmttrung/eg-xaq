# 08 — Danh sách câu hỏi cho chị mentor

> In file này ra và hỏi theo danh sách. Mỗi câu hỏi tương ứng với một chỗ trống cụ thể
> trong code — không phải hỏi cho biết.
>
> Chi tiết kỹ thuật đầy đủ: `docs/02-data-contract.md`.

---

## A. Ưu tiên cao nhất — không có thì không nối được (7 câu)

| # | Câu hỏi | Vì sao cần |
|---|---|---|
| **A1** | Em xin **10 file checkpoint** và quy ước đặt tên được không? (`model_step0.pkl`… ?) | Phải chọn đúng model cho bước dự báo đang giải thích. Dùng model t+0 để giải thích dự báo t+3 là sai hoàn toàn |
| **A2** | Lúc lưu `.pkl` thì môi trường dùng **phiên bản** `python`, `xgboost`, `scikit-learn` nào? | `pickle` rất dễ vỡ giữa các phiên bản. Lệch là không mở được file |
| **A3** | Trong pickle là `XGBRegressor` (sklearn API), `xgboost.Booster`, hay một `Pipeline`? | Quyết định cách gọi `shap.TreeExplainer` |
| **A4** | Em xin **danh sách tên đặc trưng theo đúng thứ tự**, cho từng step? | ⚠ Bẫy nguy hiểm nhất: XGBoost không kiểm tra tên cột khi nhận `numpy.ndarray`. Sai thứ tự → vẫn ra số đẹp, SHAP vẫn vẽ được, và **toàn bộ kết luận sai mà không có dấu hiệu nào** |
| **A5** | **Target** có biến đổi không (log1p, chuẩn hóa)? Có **scaler** riêng không? | Nếu target là log-PM2.5 thì giá trị SHAP nằm trên thang log, không phải µg/m³. Nói sai đơn vị trong câu trả lời là lỗi nặng |
| **A6** | Em xin **code/notebook sinh đặc trưng từ GFS**? | Để tái tạo được vector đặc trưng tại một điểm bất kỳ |
| **A7** | Em xin một **mẫu dữ liệu huấn luyện** (~5.000–20.000 dòng, CSV) được không? | Hai việc: (a) background cho SHAP interventional, (b) hiệu chỉnh ngưỡng rule theo phân phối GFS thật — "PBLH bình thường ở Hà Nội là bao nhiêu" |

Nếu chỉ hỏi được **3 câu**, hỏi **A1, A4, A7**.

---

## B. Về đặc trưng — ảnh hưởng trực tiếp tới cách diễn giải (6 câu)

| # | Câu hỏi | Vì sao cần |
|---|---|---|
| **B1** | Danh sách biến GFS thô: tên, **đơn vị**, mực áp? | Kelvin vs °C, mét vs milimét — đổi sai là rule sai toàn bộ |
| **B2** | Biến accum "mean/max 2–3 ngày" tính **lùi về quá khứ** hay có bao gồm tương lai? Neo vào thời điểm nào? | Ảnh hưởng trực tiếp tới câu "gió yếu kéo dài 3 ngày" trong đầu ra |
| **B3** | Có **đặc trưng tĩnh theo pixel** không (độ cao, land use, mật độ dân, khoảng cách đường)? | ⭐ **Quan trọng cho phần nghiên cứu.** Nếu có, đó là kênh mà mô hình học "pixel này vốn bẩn" — ứng viên số 1 cho tương quan giả |
| **B4** | Có **đặc trưng thời gian** không (day-of-year, tháng, sin/cos mùa)? | Cùng lý do B3: mô hình có thể học "tháng 1 thì bẩn" thay vì học cơ chế |
| **B5** | Có dùng **PM2.5 quá khứ** (lag) làm đầu vào không? | Nếu có, SHAP ở t+0 sẽ bị persistence chiếm hết — phải xử lý riêng khi diễn giải |
| **B6** | Múi giờ **UTC hay giờ VN**? Giờ khởi tạo GFS (00/06/12/18Z)? Step là **ngày hay giờ** (t+9 = 9 ngày hay 9 giờ)? | Lệch 7 tiếng làm sai toàn bộ episode đánh giá |

> B3 và B4 không phải câu hỏi kỹ thuật thuần túy — chúng quyết định một phần kết quả
> nghiên cứu. Nếu câu trả lời là "có", khóa luận sẽ có một mục báo cáo *"X% attribution
> của mô hình đến từ đặc trưng không mang cơ chế vật lý"*, và đó là một phát hiện thật
> chứ không phải lỗi cần giấu.

---

## C. Về raster (5 câu)

| # | Câu hỏi |
|---|---|
| **C1** | CRS (EPSG:4326?), độ phân giải, bounding box của miền dự báo? |
| **C2** | Bản đồ đặc trưng đầu vào lưu định dạng gì — GeoTIFF nhiều band, NetCDF, hay `.npy`? |
| **C3** | Quy ước **band ↔ đặc trưng**? |
| **C4** | Giá trị **nodata** và mask (biển, ngoài miền)? |
| **C5** | Dữ liệu lưu ở đâu, em truy cập thế nào — ổ mạng, S3, hay chỉ có trên server PopGIS? |

C4 quan trọng hơn vẻ ngoài: một pixel biển có PBLH = 0 sẽ kích hoạt rule "lớp xáo trộn
thấp" ở mức tối đa và sinh ra một lời giải thích hoàn toàn bịa.

---

## D. Về web PopGIS (3 câu)

| # | Câu hỏi | Vì sao cần |
|---|---|---|
| **D1** | Web có **API lấy giá trị dự báo tại một điểm** không? | Nếu có → có thể **không cần chạy lại mô hình**, chỉ cần SHAP. Đơn giản hóa rất nhiều và né được rủi ro lệch phiên bản pickle |
| **D2** | Quy đổi PM2.5 → AQI đang dùng thang **US EPA** hay **VN AQI (QĐ 1459/QĐ-TCMT)**? | Hai thang lệch nhau đáng kể. Nếu hệ thống của em nói AQI khác web thì người dùng thấy mâu thuẫn |
| **D3** | Có log truy vấn người dùng không? | Nguồn tốt để lấy **câu hỏi thật** cho bộ đánh giá, thay vì em tự nghĩ ra |

---

## E. Về đánh giá (3 câu)

| # | Câu hỏi | Vì sao cần |
|---|---|---|
| **E1** | Mô hình huấn luyện trên khoảng thời gian nào? Tập test là những ngày nào? | ⚠ Episode đánh giá **phải nằm ngoài tập huấn luyện**, nếu không SHAP đang giải thích một dự báo mà mô hình đã "nhìn thấy đáp án" |
| **E2** | Chỉ số đánh giá của mô hình (RMSE/MAE/R²) theo từng step? | Khóa luận phải nói rõ: "hệ thống này giải thích một mô hình có chất lượng X" |
| **E3** | Chị có biết tài liệu/báo cáo nào phân tích nguyên nhân các đợt ô nhiễm ở Hà Nội không? (CEM, báo cáo dự án…) | Nguồn gán nhãn nguyên nhân cho bộ đánh giá — phần tốn công nhất |

---

## F. Câu hỏi phương án dự phòng (hỏi nếu artifacts chưa sẵn sàng)

1. Nếu chưa lấy được checkpoint, chị **chạy hộ SHAP theo lô** cho khoảng 50 ngày rồi
   xuất CSV được không? Schema em cần: `date, step, feature, value, shap` cộng
   `base_value, prediction` mỗi dòng. (Xem `docs/02-data-contract.md §7.2`.)
   → Cách này **không cần** em có checkpoint, không cần khớp phiên bản thư viện.

2. Nếu không được nữa, em dùng tạm **ERA5** (em đã tải sẵn 2017–2024 cho hộp Hà Nội)
   và tự train một XGBoost đơn giản để có đối tượng giải thích. Phần "tái sử dụng mô
   hình lab" chuyển thành "đã thiết kế sẵn adapter + test hợp đồng". Chị thấy hướng
   này có ổn không ạ?

---

## G. Trả lời rồi thì làm gì

Điền câu trả lời vào `docs/02-data-contract.md` (đánh dấu ✅ từng mục), rồi theo
§8 của file đó:

```
1. Cập nhật knowledge/feature_map.yaml theo A4     ← LÀM ĐẦU TIÊN
2. Viết src/data/lab_raster.py
3. Viết src/xai/lab_xgb.py
4. python scripts/calibrate_thresholds.py --csv <mẫu từ A7>
5. pytest                                          ← test lõi phải VẪN XANH
6. Đổi .env: EGXAQ_DATA_BACKEND=lab, EGXAQ_XAI_BACKEND=lab
```

Bước 1 là bước dễ quên nhất và hậu quả im lặng nhất: không cập nhật feature map thì
`shap_agreement` luôn bằng 0 và toàn bộ consistency check trở nên vô nghĩa — mà hệ
thống vẫn chạy, vẫn ra câu trả lời trông rất hợp lý.

Bước 5 là điểm kiểm soát: nếu test lõi vỡ khi cắm data thật, nghĩa là lõi đã lỡ phụ
thuộc vào mock ở đâu đó → phải sửa ngay, **không được nới test**.
