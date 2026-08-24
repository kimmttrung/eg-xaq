# 04 — Cơ chế suy luận & Consistency Scoring

> File liên quan: `src/reasoning/` (`rules.py`, `hypotheses.py`, `consistency.py`,
> `scoring.py`, `engine.py`, `derive.py`)

## 1. Vì sao chọn Hybrid, không chọn thuần Bayesian/Causal

| Phương án | Ưu | Nhược | Quyết định |
|---|---|---|---|
| Rule-based | Minh bạch, nhanh, không cần data lớn | Cứng nhắc, không xử lý bất định | ✅ **Lõi** |
| Probabilistic scoring | Xếp hạng được giả thuyết | Cần hiệu chỉnh trọng số | ✅ **Bổ trợ** |
| SHAP attribution | Gắn trực tiếp với mô hình, định lượng | Giải thích mô hình ≠ thực tại | ✅ **Bắt buộc, có đối chiếu** |
| Graph reasoning (KG) | Suy luận nhiều bước, truy vết tốt | Xây thủ công | ✅ **Lõi** |
| Bayesian Network | Đẹp về lý thuyết | Cần học cấu trúc + tham số, dữ liệu lớn | ⚠️ Future work |
| SCM / do-calculus | Nhân quả "thật" | Giả định căn chỉnh rất mạnh, khó validate | ⚠️ Future work |

**Lý do loại Bayesian/SCM khỏi phạm vi**: chúng đòi hỏi dữ liệu lớn, giả định nhân
quả mạnh, và rất khó validate trong khung thời gian khóa luận. Làm hời hợt sẽ tạo
**ảo giác về tính nhân quả** — tệ hơn là không làm. Hybrid rule + SHAP + RAG cho độ
tin cậy *kiểm chứng được*, và vẫn có tính mới ở phần consistency scoring.

---

## 2. Luồng suy luận

```
Observation ──► derive_features ──► variable_lookup ──► RuleEngine
                                                            │
                                                    RuleFiring × N
                                                            │
                                        build_hypotheses ◄──┘
                                                            │
Attribution ──► annotate_contributions ──► assess_consistency
                                                            │
                                              Hypothesis + verdict
                                                            │
                              GatedRetriever.attach ◄───────┘   (tầng RAG)
                                                            │
                                        score_hypotheses ◄──┘
                                                            │
                                          causes, suppressors
```

`ReasoningEngine` cố ý tách làm **hai giai đoạn** — `reason()` rồi `rank()` — với chỗ
trống ở giữa cho RAG. Lý do: RAG là tầng I/O (mạng, vector DB) mà `reasoning/` không
được phép phụ thuộc. Nhờ vậy toàn bộ lõi test được offline, và cấu hình ablation D
(tắt RAG) chỉ đơn giản là bỏ qua bước ở giữa.

---

## 3. Chỉ số dẫn xuất

| Chỉ số | Công thức | Ghi chú thiết kế |
|---|---|---|
| `lapse_rate_c_per_km` | `(t2m − t850) / 1.5` | Âm = nghịch nhiệt. Bình thường ≈ 6.5 |
| `ventilation_index_m2s` | `blh × wind_speed` (**cùng ngày**) | Xem bên dưới |
| `blh_anomaly_pct` | `(blh − clim) / clim × 100` | Anomaly thuyết phục hơn giá trị tuyệt đối |
| `stagnation_days` | Suy từ biến accum 2/3 ngày | Xấp xỉ — xem bên dưới |
| `aqi_vn` | Bảng QĐ 1459/QĐ-TCMT | ⚠ Phải khớp thang mà PopGIS dùng (W2) |

**Vì sao ventilation index dùng giá trị cùng ngày, không dùng cửa sổ tích lũy:**
ghép PBLH *cực tiểu 2 ngày* (một thống kê ban đêm, rất thấp) với gió *trung bình 2
ngày* hạ thấp chỉ số một cách hệ thống — ngày trong lành cũng bị xếp là "kém thông
thoáng". Khía cạnh cộng dồn nhiều ngày đã có `stagnation_days` (R10) và cơ chế
`MECH_MULTIDAY_ACCUMULATION` lo; nhồi vào đây là **đếm hai lần**.

**`stagnation_days` hiện là xấp xỉ**, không phải phép đếm thật: ta chỉ có giá trị
trung bình cửa sổ, không có chuỗi ngày. Khi có dữ liệu chuỗi đầy đủ từ lab (F3 trong
data contract), thay bằng phép đếm thật.

---

## 4. Consistency Scoring — đóng góp nghiên cứu số 2

### 4.1. Công thức

Với mỗi cơ chế `h`:

```
shap_agreement(h) = Σ_v  share(v) · sign_match(v)

    share(v)      = |shap_v| / Σ|shap|                  ∈ [0, 1]
    sign_match(v) = +1  nếu dấu SHAP đúng như expected_shap
                    −1  nếu ngược
                    +1  nếu expected_shap = 'any'
```

Vì `Σ share ≤ 1`, kết quả tự nhiên nằm trong **[−1, 1]** và mang **hai thông tin cùng lúc**:

- **Độ phủ** — các biến của cơ chế chiếm bao nhiêu phần attribution của mô hình
- **Hướng** — đúng hay ngược chiều vật lý

Đây là điểm tinh tế: một chỉ số chỉ đo độ phủ sẽ không phân biệt được "mô hình đồng ý"
với "mô hình phản đối mạnh mẽ" — mà đó chính là thứ cần đo.

### 4.2. Bảng verdict

| `rule_strength` | `shap_agreement` | Verdict | Ý nghĩa |
|---|---|---|---|
| > 0.05 | ≥ +0.15 | **CONFIRMED** | Vật lý và mô hình đồng thuận |
| > 0.05 | (−0.10, +0.15) | **PARTIAL** | Mô hình không nói gì rõ ràng |
| > 0.05 | ≤ −0.10 | **CONFLICT** ⚠ | Mô hình đi ngược tri thức vật lý |
| ≤ 0.05 | ≥ +0.25 | **MODEL_ONLY** ⚠ | Mô hình dựa vào biến mà điều kiện không kích hoạt |
| ≤ 0.05 | < +0.25 | **INACTIVE** | Loại khỏi câu trả lời |
| bất kỳ | không có SHAP | **NO_SHAP** | Ablation A–B, hoặc chưa có model |

Ngưỡng CONFLICT (−0.10) **nhạy hơn** ngưỡng CONFIRMED (+0.15) một cách có chủ đích:
mâu thuẫn là tín hiệu cần biết sớm, thà báo thừa còn hơn bỏ sót.

### 4.3. CONFLICT và MODEL_ONLY không được nuốt lặng

Cả hai đều sinh cảnh báo tiếng Việt viết sẵn, đẩy lên `bundle.conflicts`, và narrator
**bắt buộc** trình bày thành mục "Điểm bất định" riêng (quy tắc 4 trong system prompt).

Đây là dữ liệu nghiên cứu:
- **CONFLICT** → dấu hiệu mô hình học tương quan giả cho nhóm biến đó.
- **MODEL_ONLY** → mô hình nhấn mạnh một cơ chế mà điều kiện khí tượng không hề kích hoạt.

Ngoài ra, nếu `non_mechanistic_share > 30%` (toạ độ, mã ngày, biến tĩnh chiếm phần lớn
attribution) → cảnh báo riêng: *"dự báo có thể đúng nhờ ghi nhớ mẫu không gian/thời
gian thay vì học cơ chế"*.

---

## 5. Chấm điểm và xếp hạng

```
score(h) = (α · rule_strength + β · shap_support + γ · rag_support) · prior_weight(h)

    shap_support   = max(0, shap_agreement)        ← chỉ phần đồng thuận DƯƠNG cộng điểm
    prior_weight   = 0.6 + 0.4 · confidence_prior  ← prior chỉ điều chỉnh, không quyết định
    α = 0.45,  β = 0.35,  γ = 0.20   (α + β + γ = 1, có assert)
```

**Vì sao α > β**: rule mã hóa tri thức vật lý đã được kiểm chứng, còn SHAP chỉ phản
ánh mô hình — vốn có thể học tương quan giả (INV-2). Cho SHAP trọng số cao hơn rule
là **mâu thuẫn với chính lập luận trung tâm của đề tài**.

**Vì sao `shap_support = max(0, agreement)`** thay vì dùng thẳng agreement: phần mâu
thuẫn được xử lý bằng **hệ số phạt riêng**, để không có chuyện hai sai số triệt tiêu
nhau thành một điểm số trung tính vô nghĩa.

**Vì sao `prior_weight ∈ [0.6, 1.0]`** thay vì nhân thẳng `confidence_prior`: nếu nhân
thẳng, một cơ chế prior 0.9 **không hề được kích hoạt** vẫn có thể vượt một cơ chế
prior 0.6 đang kích hoạt mạnh — sai về logic.

### Hệ số phạt

| Verdict | Hệ số | Vì sao vẫn giữ trong câu trả lời |
|---|---|---|
| CONFLICT | × 0.50 | Nó là điểm bất định cần **nêu ra**, không phải cần giấu đi |
| MODEL_ONLY | × 0.35 | Người đọc cần biết mô hình đang dựa vào gì, kèm cảnh báo |

Có unit test bảo vệ điều này: `test_conflict_is_penalised_but_kept`.

### Ngưỡng loại

`score < 0.08` → không đưa vào câu trả lời. Mục đích: tránh câu trả lời loãng vì
một tá cơ chế kích hoạt yếu ớt.

---

## 6. Độ tin cậy

### 6.1. Từng giả thuyết

```
đếm số nguồn đồng thuận:
    rule_strength > 0.05      → +1
    shap_agreement ≥ 0.15     → +1
    rag_support > 0           → +1

3 nguồn → CAO   |   2 → TRUNG BÌNH   |   1 → THẤP   |   0 → KHÔNG ĐỦ CĂN CỨ
verdict CONFLICT hoặc MODEL_ONLY → ép về THẤP bất kể số nguồn
```

### 6.2. Toàn câu trả lời

Lấy độ tin cậy của giả thuyết mạnh nhất, rồi **hạ một bậc** nếu:
- có mâu thuẫn (`conflicts` không rỗng), **hoặc**
- tầng RAG bị tắt (cấu hình ablation A–D)

Quy tắc thận trọng có chủ đích: một câu trả lời không có trích dẫn khoa học **không
được** mang nhãn tin cậy CAO, dù rule và SHAP có đồng thuận đến đâu.

### 6.3. Vì sao backend chấm điểm, không phải LLM

Nếu để LLM tự đánh giá độ tin cậy, nó sẽ đánh giá **theo độ trôi chảy của văn bản
nó vừa viết**, không theo bằng chứng. Confidence được tính ở backend và narrator chỉ
được **chép lại** — quy tắc 3 trong system prompt.

---

## 7. Hướng mở rộng (future work)

Thêm một mô-đun **causal discovery** nhỏ trên chuỗi thời gian trạm quan trắc để kiểm
định các cạnh KG bằng dữ liệu — biến KG thủ công thành **KG có bằng chứng thống kê**.
Đóng góp đẹp nhưng không bắt buộc cho bản khóa luận; nêu trong chương kết luận.
