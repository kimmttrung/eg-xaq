# EG-XAQ

**Evidence-Grounded Explainable Air Quality Causal System** — hệ thống AI giải thích nguyên
nhân ô nhiễm PM2.5 dựa trên bằng chứng.

> Khóa luận tốt nghiệp. Không phải chatbot dự báo PM2.5 — mà là hệ thống *giải thích* đầu ra
> của mô hình dự báo có sẵn, bằng một chuỗi bằng chứng truy vết được.

---

## Vấn đề

Hỏi ChatGPT *"vì sao Hà Nội hôm nay ô nhiễm?"* sẽ nhận được một câu trả lời trôi chảy, hợp lý,
và **không kiểm chứng được**. Nó không biết PBLH hôm nay là bao nhiêu, không biết mô hình dự
báo đang dựa vào biến nào, và không trích dẫn được bài báo nào.

EG-XAQ trả lời cùng câu hỏi đó bằng bốn lớp bằng chứng ghép lại:

| Lớp | Nguồn | Trả lời câu hỏi |
|---|---|---|
| **Dữ liệu** | GFS/ERA5 + FIRMS tại (X, T) | Điều kiện khí quyển thực tế là gì? |
| **Mô hình** | SHAP trên XGBoost của lab | Mô hình dự báo dựa vào biến nào? |
| **Cơ chế** | Rule Base + Knowledge Graph | Điều kiện đó kích hoạt cơ chế vật lý nào? |
| **Tài liệu** | Scientific RAG (Qdrant) | Khoa học nói gì về cơ chế đó? |

LLM chỉ đóng vai **constrained narrator** — diễn đạt lại bằng chứng, tuyệt đối không tự suy đoán.

---

## Ba bất biến của hệ thống

Đây là những ràng buộc thiết kế, không phải khuyến nghị. Vi phạm là hỏng luận điểm của đề tài.

**INV-1 — LLM không bao giờ là nguồn của nguyên nhân.**
LLM nhận `EvidenceBundle` (JSON) và diễn đạt lại. Không được thêm bất kỳ cơ chế, con số hay
trích dẫn nào không có trong bundle.

**INV-2 — SHAP giải thích MÔ HÌNH, không giải thích THỰC TẠI.**
*"Vì sao mô hình dự báo cao?"* và *"vì sao không khí thực tế xấu?"* là hai câu hỏi khác nhau về
tri thức luận. Nếu mô hình học phải tương quan giả, SHAP vẫn "trung thực" báo cáo lý do **sai**
của mô hình. Vì vậy attribution luôn phải đối chiếu với tri thức vật lý trước khi ra tới người dùng.

**INV-3 — Fail-safe: thiếu bằng chứng thì nói "không đủ căn cứ".**
Dữ liệu thiếu nghĩa là cơ chế liên quan **chưa được kiểm tra**, không phải đã bị loại trừ. Hệ
thống nêu rõ điều đó thay vì đoán bù.

---

## Ví dụ đầu ra

```
**Kết luận**
PM2.5 ≈ 95 µg/m³ (AQI 162 — Xấu) tại Hà Nội ngày 2024-01-15 (dự báo t+0).
Nguyên nhân chủ đạo: Thông thoáng khí quyển kém [R7, R10]. Chỉ số thông gió
(PBLH × gió) chỉ 256 m²/s, dưới ngưỡng 4000 m²/s.
Cơ chế góp phần: nghịch nhiệt giữ ô nhiễm, tích tụ cộng dồn nhiều ngày.

**Bằng chứng dữ liệu**
- [D2] Chiều cao lớp xáo trộn (PBLH): 320 m — thấp hơn trung bình khí hậu 61%
- [D3] Tốc độ gió 10 m: 0.8 m/s
- [D13] Gradient nhiệt theo độ cao: -1.13 °C/km — bình thường ≈ 6.5; ≤0 là nghịch nhiệt

**Bằng chứng từ mô hình**
_Các giá trị dưới đây giải thích DỰ BÁO CỦA MÔ HÌNH, không phải nhân quả thực tế._
- [S2] `blh_min_2d` ↑ +7.96 µg/m³
- Tỉ lệ attribution từ đặc trưng không mang cơ chế: 7%

**Cơ chế khoa học**
- Nghịch nhiệt giữ ô nhiễm (điểm 0.51, CONFIRMED, tin cậy TRUNG BÌNH)
  _Dấu vết luật:_ R3: lapse_rate = -1.13 < 2 °C/km; R1: blh_m = 320 < 500 m
  - [E1] Nguyen et al. (2022) — ... doi:10.xxxx

**Độ tin cậy & hạn chế**
- Độ tin cậy tổng thể: TRUNG BÌNH
- Không kiểm tra được 'Gió từ hướng có nguồn thải' (thiếu dữ liệu:
  source_sector_alignment) → chưa thể khẳng định hay loại trừ: vận chuyển từ cụm nguồn.
```

---

## Đóng góp nghiên cứu

1. **Kiến trúc tích hợp** attribution mô hình + KG cơ chế + Scientific RAG thành một pipeline
   giải thích có trích dẫn cho bối cảnh đô thị Việt Nam.
2. **Consistency scoring** giữa SHAP (thống kê) và tri thức khí quyển (vật lý) → một chiều
   faithfulness mới, đồng thời là cơ chế sinh confidence. Bốn verdict:
   `CONFIRMED` / `PARTIAL` / `CONFLICT` / `MODEL_ONLY`.
3. **Bộ đánh giá + ablation** (A–E) định lượng chứng minh mỗi tầng grounding làm giảm
   hallucination.

---

## Chạy thử trong 3 phút

Không cần data lab, không cần API key, không cần Docker.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python scripts/demo_explain.py --episode winter_inversion
python scripts/demo_explain.py --list
pytest
```

Vài thứ đáng xem:

```powershell
# Ngày trong lành — hệ thống giải thích được cả "vì sao SẠCH"
python scripts/demo_explain.py --episode cold_surge_clean

# Mô phỏng mô hình học tương quan giả → sinh verdict CONFLICT
python scripts/demo_explain.py --episode winter_inversion --xai-mode conflicting

# Ablation: tắt dần từng tầng bằng chứng
python scripts/demo_explain.py --episode winter_inversion --ablation B

# Xuất sơ đồ Knowledge Graph để dán vào khóa luận
python scripts/demo_explain.py --episode winter_inversion --mermaid
```

Demo dùng `MockObservationProvider` — dữ liệu tổng hợp mô phỏng các kịch bản điển hình của
Hà Nội. Khi có checkpoint XGBoost + raster GFS từ lab, chỉ cần viết provider tương ứng và đổi
`EGXAQ_DATA_BACKEND=lab`; lõi suy luận, RAG và narrator **không phải sửa một dòng nào**.

## Bật Scientific RAG

```powershell
docker compose up -d qdrant            # http://localhost:6333/dashboard
pip install -r requirements-embed.txt
python scripts/build_corpus.py --per-query 20
python scripts/index_corpus.py --embedding hf
```

---

## Bố cục

```
eg-xaq/
├── docs/                 ← thiết kế chi tiết, đọc trước khi sửa module tương ứng
├── knowledge/            ← TRI THỨC: rules.yaml, mechanisms.yaml, feature_map.yaml
├── src/
│   ├── config.py  schemas.py  kb.py  pipeline.py  geoutils.py
│   ├── data/             ← ObservationProvider (mock | lab)
│   ├── xai/              ← AttributionProvider (mock | lab XGBoost+SHAP)
│   ├── reasoning/        ← rule engine, KG, consistency, scoring  ← lõi, không có I/O
│   ├── rag/              ← corpus, chunk, embed, Qdrant, gated retrieval
│   └── narrator/         ← prompt ràng buộc + Claude API
├── scripts/              ← demo_explain, build_corpus, index_corpus, calibrate_thresholds
└── tests/                ← 98 test, chạy offline
```

**Quy tắc kiến trúc**: `reasoning/` không được import `xgboost`, `shap`, `rasterio`,
`qdrant_client` hay `anthropic`. Nhờ vậy phần *ăn điểm* của khóa luận test được mà không cần
hạ tầng ngoài.

**Tri thức không nằm trong code**: muốn đổi ngưỡng PBLH thì sửa `knowledge/rules.yaml`, không
sửa Python. Mỗi ngưỡng bắt buộc có trường `rationale` và `source` — có unit test kiểm.

---

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/00-overview.md](docs/00-overview.md) | Bản chất bài toán, phân biệt Q1/Q2, định vị nghiên cứu |
| [docs/01-architecture.md](docs/01-architecture.md) | Pipeline, ranh giới module, cấu hình ablation |
| [docs/02-data-contract.md](docs/02-data-contract.md) | **Cần xin gì từ lab** — đọc trước khi gặp mentor |
| [docs/03-knowledge-base.md](docs/03-knowledge-base.md) | Rule, cơ chế, KG, feature map |
| [docs/04-reasoning.md](docs/04-reasoning.md) | Công thức scoring, consistency, confidence |
| [docs/05-rag.md](docs/05-rag.md) | Corpus, chunking, gated retrieval, citation |
| [docs/06-evaluation.md](docs/06-evaluation.md) | Metric, ablation, bộ ground truth |
| [docs/07-roadmap.md](docs/07-roadmap.md) | Lộ trình, rủi ro, definition of done |
| [docs/08-questions-for-mentor.md](docs/08-questions-for-mentor.md) | Danh sách câu hỏi cụ thể |

---

## Trạng thái

| Thành phần | Trạng thái |
|---|---|
| Knowledge base (13 rule, 12 cơ chế) | ✅ |
| Reasoning engine + consistency + scoring | ✅ |
| Knowledge Graph + xuất Mermaid | ✅ |
| Scientific RAG (Qdrant, gated retrieval) | ✅ khung — corpus còn rỗng |
| Constrained narrator (dryrun + Claude API) | ✅ |
| Pipeline + ablation A–E | ✅ |
| **Data GFS + checkpoint XGBoost của lab** | ❌ **chờ mentor** |
| Đánh giá / ablation định lượng | ⬜ |

---

## Giấy phép & bản quyền tài liệu

Corpus RAG chỉ lưu **metadata + abstract** cho tài liệu đóng; full-text **chỉ với bài
open-access** (xác định qua OpenAlex `is_oa` / Unpaywall). Ràng buộc này được thực thi bằng
code trong `src/rag/models.py`, có unit test. Xem [docs/05-rag.md](docs/05-rag.md) §1.
