# 00 — Bản chất bài toán & định vị nghiên cứu

## 1. Đây là bài toán gì?

Không phải QA thuần túy. Không phải RAG thông thường. Bản chất là:

> **Evidence-Grounded Causal Explanation** cho một hiện tượng môi trường không–thời gian.

Nó nằm ở giao của bốn lớp bài toán:

| Lớp | Vai trò trong đề tài |
|---|---|
| Explainable AI (XAI) | Giải thích vì sao mô hình dự báo ra PM2.5 cao (feature attribution) |
| Knowledge-Grounded / Scientific RAG | Neo mọi tuyên bố nhân quả vào tài liệu có thể trích dẫn |
| Rule-based / Causal Reasoning | Ánh xạ điều kiện quan sát → cơ chế vật lý–hóa học đã được công nhận |
| Geospatial reasoning | Suy luận theo vị trí: nguồn thải thượng nguồn gió, KCN, giao thông, cháy sinh khối |

---

## 2. Phân biệt cốt lõi: Q1 ≠ Q2

Đây là điểm tri thức luận quan trọng nhất của cả khóa luận. Phải tách bạch:

| | Q1 | Q2 |
|---|---|---|
| Câu hỏi | "Vì sao **mô hình** dự báo PM2.5 cao?" | "Vì sao **không khí thực tế** xấu?" |
| Nói về | Mô hình | Thế giới thực |
| Trả lời bằng | SHAP / feature attribution | Điều kiện quan sát + cơ chế khoa học + tài liệu |
| Sai lầm nếu nhầm | Đưa SHAP ra như "nguyên nhân ô nhiễm" | — |

**Vì sao không được đánh đồng?** Nếu mô hình học phải tương quan giả (spurious correlation) —
ví dụ mô hình học rằng "tháng 1 thì PM2.5 cao" mà không thực sự học cơ chế nghịch nhiệt — SHAP
vẫn *trung thực* báo cáo lý do (sai) của mô hình. SHAP faithful với mô hình, không faithful với
thực tại.

Với mô hình của lab, rủi ro này **cụ thể và có thật**: mô hình dự báo per-pixel trên toàn raster,
nên nó rất dễ học các mẫu không gian (pixel này luôn bẩn hơn pixel kia) thay vì cơ chế khí tượng.
Consistency check chính là công cụ phát hiện điều đó.

---

## 3. Chính từ chỗ đó sinh ra tính mới

```
        Rule/KG nói gì            SHAP nói gì              Kết luận
        (vật lý khí quyển)        (thống kê từ mô hình)
        ────────────────────────────────────────────────────────────────────
   (a)  Kích hoạt mạnh       ✓    Top contributor, cùng dấu   → CONFIRMED, confidence CAO
   (b)  Kích hoạt mạnh       ✓    Không xuất hiện             → PARTIAL, confidence TRUNG BÌNH
   (c)  Không kích hoạt      ✗    Top contributor             → MODEL_ONLY, cần điều tra
   (d)  Kích hoạt mạnh       ✓    Top contributor, NGƯỢC dấu  → CONFLICT ⚠ (phát hiện đáng giá)
```

Ô **(c)** và **(d)** là những trường hợp hiếm được báo cáo trong tài liệu XAI cho chất lượng
không khí. Chúng vừa là cơ chế sinh confidence, vừa là một **chiều faithfulness mới**:
*attribution của mô hình có nhất quán với tri thức vật lý không?*

→ Đây là đóng góp có thể công bố, và là câu trả lời cho câu hỏi chắc chắn hội đồng sẽ hỏi:
*"Hệ thống của em khác gì ChatGPT?"*

---

## 4. Chuỗi giải thích lý tưởng

```
Điều kiện quan sát (data) ─┐
                           ├─► Cơ chế (Rule/KG) ─► Tài liệu (RAG) ─► Câu trả lời có trích dẫn
Attribution mô hình (SHAP)─┘        ▲                    ▲
                                    └──── đối chiếu ─────┘  → sinh confidence
```

Mọi câu trong đầu ra phải truy vết được về một `evidence_id` cụ thể. Câu nào không truy vết
được là hallucination — và tỉ lệ đó là một metric trong chương đánh giá.

---

## 5. Định vị so với các hướng hiện có

| Hướng | Điểm mạnh | Vì sao chưa đủ cho đề tài này |
|---|---|---|
| XAI cho AQ (SHAP trên XGBoost/LSTM) | Định lượng đóng góp đặc trưng | Chỉ trả lời Q1; không có cơ chế, không trích dẫn |
| Source apportionment (PMF, CMB) | Chính xác về nguồn | Cần dữ liệu thành phần hóa học; không có ở VN quy mô này |
| Mô hình vận chuyển hóa học (CMAQ, WRF-Chem) | Nhân quả vật lý đầy đủ | Rất nặng, ngoài tầm khóa luận |
| RAG / Scientific RAG | Trích dẫn được, chống bịa | Không biết gì về điều kiện thực tế tại (X, T) |
| Causal ML (SCM, Bayesian network) | Nhân quả "thật" | Cần giả định mạnh + dữ liệu lớn, không validate nổi trong 6 tháng |

**Khoảng trống (research gap)**: chưa có nhiều hệ thống tích hợp **đồng thời** attribution mô
hình + KG cơ chế + Scientific RAG thành một pipeline giải thích có trích dẫn, có đo
faithfulness/citation, cho bối cảnh đô thị Việt Nam.

---

## 6. Ba đóng góp tuyên bố

1. **Kiến trúc EG-XAQ** — pipeline tích hợp Data → SHAP → Rule/KG → Scientific RAG → constrained
   narrator, mọi tuyên bố truy vết được.
2. **Consistency scoring** giữa attribution thống kê và tri thức vật lý → chiều faithfulness mới
   + cơ chế sinh confidence có hiệu chỉnh (calibrated).
3. **Bộ đánh giá & ablation** trên các episode ô nhiễm thực tế của Hà Nội, định lượng chứng minh
   mỗi tầng grounding làm giảm hallucination.

---

## 7. Phạm vi: cái gì KHÔNG làm

Khóa cứng để kịp deadline. Những thứ sau xếp vào *future work*, nêu trong khóa luận nhưng không
triển khai:

- Bayesian Network / SCM / do-calculus (chỉ nhắc như hướng mở rộng).
- Sentinel-5P (TROPOMI NO₂/SO₂) — nặng, mây che, xử lý ảnh vệ tinh tốn thời gian.
- Traffic real-time (API đắt) → dùng proxy tĩnh từ OSM road network + giờ cao điểm.
- Huấn luyện lại mô hình dự báo — **tái sử dụng nguyên checkpoint của lab**.
- Causal discovery để kiểm định cạnh KG bằng dữ liệu — đẹp nhưng không bắt buộc.
