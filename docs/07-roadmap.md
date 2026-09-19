# 07 — Lộ trình & Definition of Done

## 1. Trạng thái hiện tại (cập nhật 2026-09-17)

| Giai đoạn | Hạng mục | Trạng thái |
|---|---|---|
| **G0** | Scaffold, docs, tooling, Docker Compose | ✅ |
| **G0** | Knowledge base (13 rule, 12 cơ chế, feature map) | ✅ |
| **G0** | Reasoning engine + consistency + scoring | ✅ |
| **G0** | Knowledge Graph + xuất Mermaid | ✅ |
| **G0** | Constrained narrator (dryrun + Claude API) | ✅ |
| **G0** | Pipeline + cấu hình ablation A–E | ✅ |
| **G1** | Corpus thật: **1025 bài → 1101 chunk**, embed BGE-M3 trên Kaggle T4, nạp Qdrant Cloud | ✅ |
| **G1** | Kiểm soát bản quyền theo **giấy phép**, snowball theo tham khảo, nhãn trích dẫn theo bài | ✅ |
| **G1** | Cổng RAG `min_score = 0.569` hiệu chỉnh trên corpus thật | ✅ sơ bộ — chưa có bộ query gán nhãn |
| **G1** | Demo có trích dẫn thật (`demo_explain.py --rag qdrant`) | ✅ |
| **G1** | Đo độ phủ địa lý corpus: **Việt Nam + ĐNÁ = 8%** (83/1025) | ✅ đã đo |
| **G1** | Hàng rào `feature_map` + sửa rò rỉ INV-3 ở `stagnation_days` | ✅ |
| **G1** | Test tự động | ✅ 179, chạy offline |
| **G2** | Nối data + model của lab | ❌ **chờ mentor** — hàng rào kiểm tra đã sẵn |
| **G3** | Hiệu chỉnh ngưỡng rule theo GFS | ❌ chờ M8 |
| **G4** | Khung chấm điểm + phiếu 45 ngày + backend `era5` | ✅ khung xong |
| **G4** | Gán nhãn 45 ngày + κ + chấm số thật + ablation | ⬜ **đang làm — đường găng hiện tại** |
| **G5** | Giao diện demo (bản đồ + hỏi đáp) | ⬜ |
| **G6** | Viết khóa luận + (tùy chọn) paper | ⬜ |

### 1.1. Nhật ký theo commit

| Commit | Làm được gì |
|---|---|
| `f9ea5e3` feat(xai) | `stagnation_days` trả `None` khi không có dữ liệu gió (trước đó sinh bằng chứng "0 ngày" từ hư không). Thêm `src/xai/coverage.py` + `scripts/check_feature_map.py`: phát hiện khi tên đặc trưng của lab không ánh xạ được. Thử nghiệm cho thấy lệch tên làm cả 7 cơ chế rơi về PARTIAL và số mâu thuẫn phát hiện được tụt **8 → 0**, không có lỗi nào hiện ra |
| `13f4633` feat(rag) | Chỉ lưu toàn văn khi có giấy phép mở tường minh (bronze OA đọc được nhưng không được lưu). Snowball theo danh mục tham khảo có cổng lọc miền. Một bài = một nhãn `[E#]`, `rag_support` đếm theo bài. `PrecomputedQueryEmbedder` để máy local không cần torch |
| `a0e0830` feat(kaggle) | `notebooks/kaggle_build_kb.py` dựng KB trọn gói trên GPU, gọi chính code của repo. `demo_explain.py --rag` bật tầng RAG |
| `b8e5da0` feat(rag) | Cổng `min_score = 0.569` hiệu chỉnh trên corpus thật; `calibrate_rag_gate.py`, `corpus_coverage.py` |
| `3bd6348` docs | Cập nhật roadmap và tài liệu RAG sau khi dựng corpus |
| *chưa commit* | Bộ đánh giá (`src/evaluation.py`), chọn ngày + phiếu gán nhãn, bộ chấm A–E, κ; backend `era5` + script tải t850/áp suất/FIRMS; narrator dùng PM2.5 quan trắc khi không có dự báo |

---

## 2. Việc làm ngay được, KHÔNG cần chờ lab

Phần *ăn điểm* của khóa luận không nằm ở data, mà ở kiến trúc và đánh giá.

| # | Việc | Ước lượng | Chặn bởi | Trạng thái |
|---|---|---|---|---|
| 1 | Corpus thật + index BGE-M3 | — | — | ✅ |
| 2 | Nối `GatedRetriever` vào demo, chạy có trích dẫn | — | — | ✅ |
| 3 | Hiệu chỉnh `min_score` sơ bộ trên corpus thật | — | — | ✅ 0.569 |
| 4a | Khung chấm + phiếu 45 ngày chọn theo PM2.5 quan trắc | — | — | ✅ |
| 4b | **Gán nhãn 45 ngày** theo `data/eval/LABELING.md` | 1 tuần | — | ⬜ **ưu tiên số 1** |
| 4c | Người thứ hai gán 15 ngày → Cohen's κ | 2 ngày | 4b song song | ⬜ |
| 5 | Viết bộ câu hỏi tiếng Việt cho từng episode | 2 ngày | (4) | ⬜ |
| 6 | Kiểm lại `min_score` bằng đường cong precision trên query có nhãn | 2–3 ngày | — | ⬜ |
| 7 | Bổ sung tài liệu Việt Nam/ĐNÁ: thêm truy vấn địa phương, thu thập tay tạp chí trong nước | 3–5 ngày | — | ⬜ |
| 8 | Dựng khung đo faithfulness/hallucination (RAGAS hoặc NLI) | 3–4 ngày | — | ⬜ |
| 9 | Rà soát danh mục tài liệu tham khảo bằng chính RAG (dogfooding) | 2 ngày | — | ⬜ |
| 10 | Đường lui offline cho ngày bảo vệ: snapshot Qdrant Cloud → Qdrant local | nửa ngày | — | ⬜ |
| 11 | Giao diện demo tối giản (FastAPI + bản đồ) | 1 tuần | — | ⬜ |
| 12 | Tải t850 + áp suất ERA5 (`download_era5_supplement.py`) | nửa ngày | tài khoản CDS | ⬜ |
| 13 | Tải điểm cháy FIRMS cho các ngày episode (`download_firms.py`) | 1 giờ | FIRMS_MAP_KEY, sau 4a | ⬜ |
| 14 | Quyết định: điều kiện cho phép có được xếp hạng chung với nguyên nhân không (xem §6) | — | — | ⬜ trước khi chấm số thật |
| 15 | Hỏi mentor M10: mô hình lab huấn luyện trên giai đoạn nào | — | mentor | ⬜ |

**Ưu tiên (4b) → (12, 13) → (14) → chấm số thật.** Khung đã xong; việc tốn thời gian nhất
giờ là đọc báo cáo để gán nhãn. Việc 12–13 chạy song song được trong lúc gán nhãn.

---

## 3. Đường găng: dữ liệu từ lab

```
Gặp mentor → nhận artifacts → python scripts/check_feature_map.py --checkpoint-dir ...
           → cập nhật feature_map.yaml → viết 2 provider → calibrate ngưỡng
           → chạy lại test → bật backend lab → đánh giá
```

**Bước `check_feature_map.py` không được bỏ.** Ánh xạ đặc trưng hỏng không gây lỗi — nó chỉ
tắt lặng lẽ bước đối chiếu Rule↔SHAP (đóng góp nghiên cứu số 2).

Chuẩn bị trước khi gặp: đọc `docs/08-questions-for-mentor.md`, in ra, hỏi theo danh sách.

**Nếu artifacts trễ**, ba phương án dự phòng theo thứ tự ưu tiên:

1. **Dùng ERA5** (đã có sẵn ở `../era5/`, 96 tháng 2017–2024) làm nguồn khí tượng và
   tự train một XGBoost đơn giản. Khóa luận vẫn đủ nội dung; phần "tái sử dụng mô hình
   lab" chuyển thành "đã thiết kế sẵn adapter, minh chứng bằng test hợp đồng".
2. **Xin lab chạy hộ SHAP theo lô** cho ~50 episode, xuất CSV theo schema ở
   `docs/02-data-contract.md §7.2`. Không cần checkpoint, không cần khớp phiên bản.
3. **Dùng API PopGIS** (W1) cho giá trị dự báo + ERA5 cho khí tượng. Mất phần SHAP
   nhưng giữ được ba tầng còn lại.

Phương án 1 đáng chuẩn bị sớm vì dữ liệu ERA5 **đã có sẵn trên máy**.

---

## 4. Lộ trình theo tuần (tính từ 2026-09-15)

| Tuần | Mục tiêu | Sản phẩm |
|---|---|---|
| ~~0~~ | ~~Corpus + index + demo có trích dẫn~~ | ✅ xong 2026-09-12 |
| 1 | Commit phần còn lại; gặp mentor, nhận artifacts hoặc chốt phương án dự phòng | Data contract điền xong |
| 1–2 | Gán nhãn 45 episode + tải t850/áp suất/FIRMS; bổ sung tài liệu Việt Nam | `data/eval/episodes.yaml` |
| 2–3 | Nối provider lab + calibrate ngưỡng | Backend `lab` chạy |
| 3–4 | Khung đo faithfulness + chạy ablation A–E | Bảng kết quả chính |
| 4–5 | Metric riêng (consistency rate, conflict rate, non-mech share) | Đóng góp số 2 định lượng |
| 5–6 | Human eval (2–3 người) + κ | Bảng rubric |
| 6–7 | Giao diện demo + đường lui offline | Bản đồ + hỏi đáp chạy được |
| 7–11 | Viết khóa luận | Bản thảo |
| 11+ | (tùy chọn) Draft paper | — |

Các giai đoạn chồng lấn có chủ đích. Human eval cần người → **sắp lịch từ tuần 3**.

---

## 5. Definition of Done

Khóa luận coi là xong khi đủ **cả sáu**:

- [ ] Hệ thống trả lời được ≥5 dạng câu hỏi mẫu, mỗi câu có trích dẫn và độ tin cậy.
  *(Tiến độ: đã có trích dẫn thật trên dữ liệu mock; còn thiếu dữ liệu thật.)*
- [ ] Ablation A→E cho thấy hallucination **giảm rõ rệt**, faithfulness **tăng đơn điệu**.
- [ ] Có bảng số liệu định lượng: faithfulness, citation accuracy, Cause F1.
- [ ] Có ≥1 kết quả cho ba metric riêng của đề tài (§3.1 trong docs/06).
- [ ] Demo bản đồ + hỏi đáp chạy được trước hội đồng.
- [ ] Mọi ngưỡng trong `rules.yaml` chỉ được vào một nguồn cụ thể, và `calibrated: true`.

Mục cuối là mục dễ quên nhất và cũng là mục hội đồng dễ hỏi nhất.

---

## 6. Bảng rủi ro

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Data lab đến trễ | Chặn G2–G3 | Phương án dự phòng §3; làm hết việc ở §2 trong lúc chờ |
| Tên đặc trưng lab lệch `feature_map.yaml` | Đối chiếu Rule↔SHAP tắt **không báo lỗi** | `scripts/check_feature_map.py` + `tests/test_feature_map_guard.py` |
| Pickle không load được (lệch phiên bản) | Không chạy được SHAP | Xin `.json`/`.ubj` (M11); hoặc venv riêng cho `lab_xgb.py` |
| Mô hình per-pixel học mẫu không gian | SHAP không mang cơ chế | **Đây là kết quả nghiên cứu**, không phải lỗi — báo cáo `non_mechanistic_share` |
| GFS phân giải thô cho đô thị | Attribution không gian kém sắc | Coi khí tượng là "điều kiện vùng" — đúng bản chất. Nêu rõ hạn chế |
| Corpus lệch địa lý — **đã đo: Việt Nam + ĐNÁ 8%** | Phát biểu mang tính địa phương thiếu căn cứ | Trích dẫn chứng minh *cơ chế*, không chứng minh địa điểm. Nêu con số trong chương hạn chế; bổ sung tài liệu Việt Nam (§2 việc 7) |
| Cổng `min_score` mới hiệu chỉnh sơ bộ | Trích dẫn lạc đề lọt, hoặc chặn nhầm | Đường cong precision trên query có nhãn (§2 việc 6). Đổi embedder → phải hiệu chỉnh lại |
| Qdrant Cloud mất mạng ngày bảo vệ | Mất tầng trích dẫn khi demo | Snapshot collection → khôi phục vào Qdrant local (docker). `--rag memory` KHÔNG dùng được với backend `precomputed` |
| Phạm vi phình to | Trễ deadline | Khóa cứng: Sentinel-5P, Bayesian, traffic real-time đều là future work |
| LLM vẫn lỡ bịa | Mất tính "no speculation" | Retrieval-gating + hậu kiểm faithfulness + abstain policy |
| Gán nhãn episode trễ | Chặn Cause F1 | Đo Q-B (faithfulness) trước — không cần nhãn |
| Điều kiện cho phép (`MECH_NO_WET_REMOVAL`) xếp trên nguyên nhân chủ động | top-1 sai dù cơ chế đúng có trong câu trả lời | Đã thấy trên episode tổng hợp (2/6). Quyết định cách xếp hạng trước khi chấm số thật (§2 việc 14) |
| Episode 2022–2024 nằm trong tập train của lab (M10) | SHAP giải thích dự báo đã "thấy đáp án" | Hỏi mentor trước khi báo cáo số có SHAP. Phần rule + RAG chấm trên ERA5 không bị ảnh hưởng |
| ERA5 gom ngày theo UTC, lapse rate lấy trung bình ngày | R3 thận trọng hơn thực tế | Nêu hạn chế; file bổ sung có sẵn t850 lúc 07:00 để thử cách tính buổi sáng |

---

## 7. Cái gì KHÔNG làm (khóa cứng phạm vi)

- Bayesian Network / SCM / do-calculus — chỉ nhắc trong chương kết luận.
- Sentinel-5P (TROPOMI NO₂/SO₂) — nặng, mây che, xử lý ảnh vệ tinh tốn thời gian.
- Traffic real-time — API đắt. Dùng proxy tĩnh từ OSM road network.
- Huấn luyện lại mô hình dự báo — **tái sử dụng nguyên checkpoint của lab**.
- Causal discovery kiểm định cạnh KG — đẹp nhưng không bắt buộc.
- Mở rộng ra nhiều tỉnh — chỉ làm nếu Hà Nội đã xong hoàn toàn.
