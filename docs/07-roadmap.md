# 07 — Lộ trình & Definition of Done

## 1. Trạng thái hiện tại (2026-08-23)

| Giai đoạn | Hạng mục | Trạng thái |
|---|---|---|
| **G0** | Scaffold, docs, tooling, Docker Compose | ✅ |
| **G0** | Knowledge base (13 rule, 12 cơ chế, feature map) | ✅ |
| **G0** | Reasoning engine + consistency + scoring | ✅ |
| **G0** | Knowledge Graph + xuất Mermaid | ✅ |
| **G0** | Scientific RAG (corpus, chunk, embed, Qdrant, gated retrieval) | ✅ khung |
| **G0** | Constrained narrator (dryrun + Claude API) | ✅ |
| **G0** | Pipeline + cấu hình ablation A–E | ✅ |
| **G0** | 98 unit test, chạy offline | ✅ |
| **G1** | Corpus thật (50–100 bài) + index bằng BGE-M3 | ⬜ |
| **G2** | Nối data + model của lab | ❌ **chờ mentor** |
| **G3** | Hiệu chỉnh ngưỡng theo GFS | ❌ chờ M8 |
| **G4** | Bộ đánh giá + ablation + human eval | ⬜ |
| **G5** | Giao diện demo (bản đồ + hỏi đáp) | ⬜ |
| **G6** | Viết khóa luận + (tùy chọn) paper | ⬜ |

---

## 2. Việc làm ngay được, KHÔNG cần chờ lab

Đây là điểm mấu chốt của chiến lược "rộng trước, sâu sau": phần *ăn điểm* của khóa
luận không nằm ở data, mà ở kiến trúc và đánh giá.

| # | Việc | Ước lượng | Chặn bởi |
|---|---|---|---|
| 1 | Bật Qdrant, tải corpus, index bằng `hf` | 1–2 ngày | — |
| 2 | Nối `GatedRetriever` vào pipeline, chạy demo có trích dẫn | nửa ngày | (1) |
| 3 | Hiệu chỉnh `min_score` trên bộ query có nhãn | 2–3 ngày | (1) |
| 4 | Chọn 30–50 episode + gán nhãn nguyên nhân từ CEM/báo cáo | 1 tuần | — |
| 5 | Viết bộ câu hỏi tiếng Việt cho từng episode | 2 ngày | (4) |
| 6 | Dựng khung đo faithfulness/hallucination (RAGAS hoặc NLI) | 3–4 ngày | — |
| 7 | Rà soát danh mục tài liệu tham khảo bằng chính RAG (dogfooding) | 2 ngày | (1) |
| 8 | Giao diện demo tối giản (FastAPI + bản đồ) | 1 tuần | — |

**Ưu tiên (1) → (2) → (4).** Corpus và bộ episode là hai thứ tốn thời gian *chờ đợi*
nhiều nhất (tải, đọc, gán nhãn), nên khởi động sớm.

---

## 3. Đường găng: dữ liệu từ lab

```
Gặp mentor → nhận artifacts → cập nhật feature_map.yaml → viết 2 provider
           → calibrate ngưỡng → chạy lại test → bật backend lab → đánh giá
```

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

## 4. Lộ trình theo tuần (từ thời điểm hiện tại)

| Tuần | Mục tiêu | Sản phẩm |
|---|---|---|
| 1 | Corpus + index + demo có trích dẫn | RAG chạy end-to-end |
| 1–2 | Gặp mentor, nhận artifacts hoặc chốt phương án dự phòng | Data contract điền xong |
| 2–3 | Chọn & gán nhãn 30–50 episode | `data/eval/episodes.jsonl` |
| 3–4 | Nối provider lab + calibrate ngưỡng | Backend `lab` chạy |
| 4–5 | Khung đo faithfulness + chạy ablation A–E | Bảng kết quả chính |
| 5–6 | Metric riêng (consistency rate, conflict rate, non-mech share) | Đóng góp số 2 định lượng |
| 6–7 | Human eval (2–3 người) + κ | Bảng rubric |
| 7–8 | Giao diện demo | Bản đồ + hỏi đáp chạy được |
| 8–12 | Viết khóa luận | Bản thảo |
| 12+ | (tùy chọn) Draft paper | — |

Các giai đoạn chồng lấn có chủ đích. Human eval cần người → **sắp lịch từ tuần 4**,
đừng để đến tuần 7 mới hỏi.

---

## 5. Definition of Done

Khóa luận coi là xong khi đủ **cả sáu**:

- [ ] Hệ thống trả lời được ≥5 dạng câu hỏi mẫu, mỗi câu có trích dẫn và độ tin cậy.
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
| Data lab đến trễ | Chặn G2–G4 | Phương án dự phòng §3; làm hết việc ở §2 trong lúc chờ |
| Pickle không load được (lệch phiên bản) | Không chạy được SHAP | Xin `.json`/`.ubj` (M11); hoặc venv riêng cho `lab_xgb.py` |
| Mô hình per-pixel học mẫu không gian | SHAP không mang cơ chế | **Đây là kết quả nghiên cứu**, không phải lỗi — báo cáo `non_mechanistic_share` |
| GFS phân giải thô cho đô thị | Attribution không gian kém sắc | Coi khí tượng là "điều kiện vùng" — đúng bản chất, vì tích tụ/nghịch nhiệt vốn là hiện tượng quy mô vùng. Nêu rõ hạn chế |
| Corpus thiếu tài liệu Hà Nội | Cơ chế địa phương không có trích dẫn | Bổ sung ĐNÁ + báo cáo tỉ lệ cơ chế có trích dẫn như một kết quả |
| Phạm vi phình to | Trễ deadline | Khóa cứng: Sentinel-5P, Bayesian, traffic real-time đều là future work |
| LLM vẫn lỡ bịa | Mất tính "no speculation" | Retrieval-gating + hậu kiểm faithfulness + abstain policy |
| Gán nhãn episode trễ | Chặn Cause F1 | Đo Q-B (faithfulness) trước — không cần nhãn |

---

## 7. Cái gì KHÔNG làm (khóa cứng phạm vi)

- Bayesian Network / SCM / do-calculus — chỉ nhắc trong chương kết luận.
- Sentinel-5P (TROPOMI NO₂/SO₂) — nặng, mây che, xử lý ảnh vệ tinh tốn thời gian.
- Traffic real-time — API đắt. Dùng proxy tĩnh từ OSM road network.
- Huấn luyện lại mô hình dự báo — **tái sử dụng nguyên checkpoint của lab**.
- Causal discovery kiểm định cạnh KG — đẹp nhưng không bắt buộc.
- Mở rộng ra nhiều tỉnh — chỉ làm nếu Hà Nội đã xong hoàn toàn.
