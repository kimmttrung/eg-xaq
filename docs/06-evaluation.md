# 06 — Phương pháp đánh giá

> Chương đánh giá quyết định điểm số của khóa luận. Hội đồng sẽ không hỏi *"giao diện
> có đẹp không?"* — họ sẽ hỏi **"làm sao minh chứng hệ thống này đáng tin hơn ChatGPT?"**

Trạng thái: ⬜ **CHƯA TRIỂN KHAI**. Hạ tầng đã sẵn sàng: `PipelineConfig.ablation()`
chạy được cả 5 cấu hình từ ngày đầu, và `EvidenceBundle` giữ đủ trạng thái trung gian
để phân tích.

---

## 1. Ba câu hỏi đánh giá

| # | Câu hỏi | Cần gì | Metric |
|---|---|---|---|
| Q-A | Hệ thống có chỉ **đúng nguyên nhân** không? | Nhãn chuyên gia | Cause F1, top-k accuracy |
| Q-B | Hệ thống có **bịa** không? | Chỉ bundle + đầu ra | Faithfulness, hallucination rate, citation accuracy |
| Q-C | **Mỗi tầng** có đóng góp gì không? | Ablation A–E | Δ metric giữa các cấu hình |

Q-B và Q-C **không cần ground truth nguyên nhân** — đo được ngay cả khi việc gán nhãn
chậm trễ. Đây là lý do nên làm chúng trước.

---

## 2. Bộ dữ liệu đánh giá

### 2.1. Chọn episode

30–50 ngày tại Hà Nội, chia ba nhóm:

| Nhóm | Số ngày | Nguyên nhân kỳ vọng | Cơ chế tương ứng |
|---|---|---|---|
| Ô nhiễm nặng mùa đông | 10–15 | Nghịch nhiệt / PBLH thấp / tù đọng | `MECH_INVERSION`, `MECH_LOW_PBLH`, `MECH_POOR_VENTILATION` |
| Ô nhiễm đột xuất tháng 3–4, 9–10 | 10–15 | Đốt rơm rạ ở các tỉnh lân cận | `MECH_BIOMASS_TRANSPORT` |
| Trời trong lành | 10–15 | Gió mùa ĐB mạnh / mưa lớn rửa trôi | `MECH_ADVECTION_CLEANSING`, `MECH_WET_DEPOSITION` |

**Nhóm thứ ba không được bỏ.** Một hệ thống chỉ giải thích được ngày bẩn mới làm nửa
bài toán, và nhóm này là phép thử tốt cho việc hệ thống có "luôn tìm ra nguyên nhân ô
nhiễm" một cách máy móc hay không.

### 2.2. Nguồn nhãn nguyên nhân

Xếp theo thứ tự ưu tiên:

1. Báo cáo chất lượng không khí của **CEM** (Trung tâm Quan trắc môi trường miền Bắc)
2. Bài báo khoa học phân tích chính đợt đó
3. Bản tin dự báo chất lượng không khí, báo chí có dẫn nguồn chuyên gia
4. **Ý kiến GVHD** — với các trường hợp còn lại

Ghi rõ nguồn nhãn cho **từng episode**. Đây là điểm yếu tiềm tàng mà hội đồng sẽ soi:
nhãn không có nguồn thì mọi con số Cause F1 đều vô nghĩa.

### 2.3. Ràng buộc quan trọng

> ⚠ Episode đánh giá phải nằm **NGOÀI tập huấn luyện** của mô hình lab (M10 trong data
> contract). Nếu không, SHAP đang giải thích một dự báo mà mô hình đã "nhìn thấy đáp
> án" — kết quả không có giá trị.

### 2.4. Bộ câu hỏi

Mỗi episode gắn 2–3 câu hỏi tiếng Việt tự nhiên, ở các mức độ khác nhau:

- Trực tiếp: *"Vì sao hôm nay Hà Nội ô nhiễm?"*
- Có tiền giả định: *"Vì sao AQI xấu dù hôm nay không mưa?"*
- Phản chứng: *"Có phải do đốt rơm rạ không?"*
- Ngoài phạm vi: *"Hôm nay có nên ra ngoài tập thể dục không?"* → kiểm tra hành vi từ chối

---

## 3. Metric

| Nhóm | Chỉ số | Cách đo |
|---|---|---|
| Đúng nguyên nhân | Cause Precision / Recall / F1, top-1 & top-3 accuracy | So `mechanism_id` hệ thống đưa ra với nhãn chuyên gia |
| Faithfulness | Tỉ lệ câu được bằng chứng trích dẫn hỗ trợ | RAGAS faithfulness hoặc NLI entailment |
| Citation accuracy | Precision của trích dẫn | Kiểm tay + tự động: trích dẫn có thực sự nói điều được gán không |
| Hallucination rate | % tuyên bố không truy vết được | Đếm câu không có nhãn [D#]/[S#]/[E#]/[R#] hoặc nhãn không tồn tại |
| Retrieval | Recall@k, MRR, nDCG | Trên bộ query có nhãn tài liệu liên quan |
| Explainability | Rubric Likert 1–5 (rõ / đúng / đủ) | 2–3 người chấm |
| Đồng thuận người chấm | Cohen's / Fleiss' κ | Bắt buộc báo cáo — thiếu κ thì human eval mất giá trị |
| Calibration | Reliability diagram, ECE | Confidence CAO có thực sự đúng thường xuyên hơn THẤP không |

### 3.1. Metric riêng của đề tài

Ba chỉ số này không có trong tài liệu chuẩn — chúng đo trực tiếp đóng góp nghiên cứu số 2:

| Chỉ số | Định nghĩa | Nói lên điều gì |
|---|---|---|
| **Mechanism-consistency rate** | % giả thuyết CONFIRMED trên tổng số được kích hoạt | Mô hình có học cơ chế vật lý không |
| **Conflict rate** | % episode có ít nhất một CONFLICT | Tần suất mô hình đi ngược tri thức khí quyển |
| **Non-mechanistic attribution share** | Trung bình `non_mechanistic_share` trên toàn bộ episode | Mô hình dựa bao nhiêu vào toạ độ / mã thời gian thay vì khí tượng |

Chỉ số thứ ba đặc biệt đáng giá với mô hình per-pixel của lab. Một kết quả kiểu
*"38% attribution đến từ đặc trưng tĩnh theo pixel"* là **phát hiện**, không phải lỗi —
và nó trả lời câu hỏi hội đồng chắc chắn sẽ hỏi: *"mô hình của lab có thực sự học
khí tượng không?"*

### 3.2. Hiệu chỉnh `min_score` của cổng chặn RAG

**Đã hiệu chỉnh sơ bộ (2026-09): `min_score = 0.569`.** Làm bằng
`scripts/calibrate_rag_gate.py` trên corpus 1025 bài, embedding BGE-M3: quét nhiều mức
cổng, đặt ngay trước điểm mà số cơ chế có trích dẫn tụt nhanh, rồi đọc tiêu đề các bài
quanh mức đó để xác nhận đúng cơ chế. Giá trị cũ 0.30 thuộc backend `hash` và không mang
sang BGE-M3 được.

Bước sơ bộ chưa đủ để báo cáo như kết quả. Cần kiểm lại bằng đường cong precision:

1. Lấy ~30 truy vấn cơ chế, gán nhãn thủ công chunk nào thực sự liên quan.
2. Quét `min_score` từ 0.1 đến 0.6.
3. Chọn điểm tối ưu precision — vì trong hệ thống này precision quan trọng hơn recall.
4. Báo cáo đường cong này trong khóa luận: nó cho thấy cổng chặn là quyết định có căn
   cứ chứ không phải con số tùy tiện.

---

## 4. Ablation study

Đây là bằng chứng định lượng cho đóng góp của kiến trúc. Chạy sẵn được:

```python
PipelineConfig.ablation("A")  # ... đến "E"
```

| Cấu hình | Data | SHAP | Rule/KG | RAG | Kỳ vọng |
|---|:---:|:---:|:---:|:---:|---|
| **A** LLM-only | ✗ | ✗ | ✗ | ✗ | Hallucination cao nhất, không trích dẫn |
| **B** + Data | ✓ | ✗ | ✗ | ✗ | Số liệu đúng, nhưng cơ chế vẫn do LLM đoán |
| **C** + SHAP | ✓ | ✓ | ✗ | ✗ | Có attribution, nhưng chưa đối chiếu vật lý |
| **D** + Rule/KG | ✓ | ✓ | ✓ | ✗ | Cơ chế có căn cứ, chưa có trích dẫn |
| **E** Full EG-XAQ | ✓ | ✓ | ✓ | ✓ | Hallucination thấp nhất, faithfulness cao nhất |

**Kỳ vọng: hallucination giảm và faithfulness tăng đơn điệu từ A → E.**

Đã có unit test bảo vệ tính đơn điệu của lượng bằng chứng
(`test_evidence_grows_monotonically_with_ablation_level`), nhưng metric chất lượng thì
phải đo bằng dữ liệu thật.

### Baseline so sánh

| Baseline | Mô tả | Trả lời câu hỏi gì |
|---|---|---|
| B-0 | LLM zero-shot (Claude/GPT) tự giải thích | Grounding có đáng không? |
| B-1 | RAG thuần: hỏi → search → trả lời | Rule/KG + SHAP có đáng không? |
| B-2 | SHAP-only: đưa thẳng attribution cho người dùng | Tầng cơ chế + tài liệu có đáng không? |
| **EG-XAQ** | Hybrid đầy đủ (cấu hình E) | — |

B-2 đặc biệt quan trọng: nó chính là cách làm phổ biến trong tài liệu XAI cho chất
lượng không khí hiện nay. Vượt được B-2 là vượt được state of the art thực dụng.

### `DryRunNarrator` làm đối chứng

`DryRunNarrator` render **đúng những gì bundle chứa, không hơn một chữ**. So sánh đầu
ra của nó với đầu ra LLM cho ta một baseline sạch: **mọi khác biệt đều là thứ LLM thêm
vào** — và đó chính là thứ cần đo khi tính hallucination.

---

## 5. Rủi ro đánh giá & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Không có ground-truth nguyên nhân đủ tin cậy | Đo cả **faithfulness** (không cần ground truth). Báo cáo Cause F1 kèm nguồn nhãn từng episode |
| Nhãn chuyên gia chủ quan | ≥2 người chấm độc lập + báo cáo κ |
| Corpus RAG thiếu tài liệu Việt Nam | Bổ sung tài liệu khu vực ĐNÁ; báo cáo tỉ lệ cơ chế có/không có trích dẫn như một kết quả |
| Episode trùng tập huấn luyện | Xin M10 trước khi chọn episode |
| Bộ test quá nhỏ (n≈40) | Báo cáo khoảng tin cậy, không chỉ điểm số; dùng bootstrap CI |

---

## 6. Thứ tự triển khai đề xuất

```
1. Q-B trước (faithfulness, hallucination, citation)  ← không cần ground truth
2. Ablation A–E trên chính bộ episode đó              ← ra được biểu đồ chính của khóa luận
3. Metric riêng của đề tài (§3.1)                     ← ra được đóng góp số 2
4. Gán nhãn nguyên nhân + Q-A                         ← tốn công nhất, làm sau
5. Human eval + κ                                     ← cần người, sắp lịch sớm
6. Calibration                                        ← nếu còn thời gian
```

Lý do đảo thứ tự so với trực giác: bước 1–3 chạy được **ngay khi có data lab**, không
phụ thuộc vào việc gán nhãn — vốn là bước dễ trễ nhất.
