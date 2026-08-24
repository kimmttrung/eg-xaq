# 05 — Scientific RAG

> File liên quan: `src/rag/` (`corpus.py`, `chunking.py`, `embedding.py`, `store.py`,
> `rerank.py`, `retrieve.py`), `scripts/build_corpus.py`, `scripts/index_corpus.py`

## 1. Bản quyền — đọc trước khi làm bất cứ gì

| Loại tài liệu | Được lưu gì |
|---|---|
| Open-access (OpenAlex `is_oa: true`, hoặc qua Unpaywall) | Metadata + abstract + **full-text** |
| Không open-access | Metadata + **abstract only** |
| Google Scholar | ❌ Không có API chính thức — cào tự động vi phạm ToS. Chỉ tra thủ công |

Ràng buộc này được **thực thi bằng code**, không chỉ ghi trong tài liệu:

```python
# src/rag/models.py
@model_validator(mode="after")
def validate_licensing(self) -> Paper:
    if self.full_text and not self.is_open_access:
        raise ValueError("Corpus chỉ được lưu full-text của bài OA")
```

Có unit test: `test_closed_access_paper_cannot_hold_full_text`.

Nêu rõ điều này trong chương phương pháp của khóa luận — nó bảo vệ tính hợp lệ của
toàn bộ công trình.

---

## 2. Điểm khác biệt cốt lõi: Gated Retrieval

**RAG thông thường:**
```
câu hỏi người dùng  →  vector search  →  đưa cho LLM
```

**EG-XAQ:**
```
câu hỏi người dùng  →  dữ liệu quan sát tại (X,T)  →  rule kích hoạt cơ chế
                    →  rag_query CỦA CƠ CHẾ  →  vector search  →  cổng chặn
                    →  trích dẫn gắn vào ĐÚNG cơ chế đó
```

Ví dụ cụ thể:

| | |
|---|---|
| Người dùng gõ | *"sao hôm nay bụi thế?"* |
| Hệ thống truy vấn vector DB | `planetary boundary layer height shallow mixing layer PM2.5 accumulation urban haze episode` |

### Ba hệ quả

1. **Truy vấn bằng ngôn ngữ khoa học, không phải ngôn ngữ người dùng.** Corpus là
   tiếng Anh học thuật; truy vấn bằng câu hỏi đời thường tiếng Việt sẽ khớp rất kém.
   Đây không phải mẹo kỹ thuật — nó là hệ quả trực tiếp của kiến trúc.

2. **Trích dẫn gắn vào cơ chế, không gắn vào câu trả lời nói chung.** Narrator biết
   chính xác câu nào được chứng minh bởi tài liệu nào → trích dẫn ở **cấp câu**.

3. **Cổng chặn (gate).** Không tài liệu nào vượt `min_score` → `rag_support = 0` và
   cơ chế bị đánh dấu thiếu căn cứ. Đây là INV-3: thà nói *"chưa có tài liệu trong kho
   hỗ trợ cơ chế này"* còn hơn trích dẫn một bài không liên quan.

---

## 3. Pipeline

```mermaid
flowchart LR
    A[OpenAlex API] --> B[Paper: metadata + abstract]
    B --> C["chunk_paper()<br/>theo mục, 320 từ, chồng lấn 15%"]
    C --> D["Embedding<br/>BGE-M3 (hf) | hash (test)"]
    D --> E[(Qdrant<br/>COSINE)]
    F["rag_query của cơ chế"] --> G[Vector search top-20]
    E --> G
    G --> H["Rerank<br/>bge-reranker-v2-m3"]
    H --> I{"score ≥ min_score?"}
    I -->|có| J["Citation E1, E2, ...<br/>+ rag_support"]
    I -->|không| K["rag_support = 0<br/>'chưa có tài liệu'"]
```

---

## 4. Xây corpus

### Nguồn: OpenAlex

Vì sao OpenAlex chứ không phải Semantic Scholar hay Google Scholar:

| Nguồn | Đánh giá |
|---|---|
| **OpenAlex** | API mở, không hạn ngạch ngặt, chỉ cần email vào polite pool, có `is_oa` |
| Semantic Scholar | Tốt, nhưng cần key và hạn ngạch chặt hơn |
| Crossref | Metadata chuẩn nhưng không có abstract |
| Google Scholar | ❌ Không có API chính thức |

**Abstract dạng inverted index.** OpenAlex trả abstract dưới dạng
`{"Low": [0], "boundary": [1], ...}` để lách bản quyền. `_reconstruct_abstract()`
dựng lại thành văn xuôi.

### Truy vấn = chính `rag_query` của các cơ chế

```powershell
python scripts/build_corpus.py --per-query 20
```

Script lấy `rag_query` của **từng cơ chế** trong `mechanisms.yaml`, cộng thêm 7 truy
vấn nền (Hà Nội, đồng bằng sông Hồng, ĐNÁ, PMF, SHAP cho AQ, gió mùa ĐB, haze Đông Á).

Nhờ vậy corpus được xây **đúng theo nhu cầu của reasoning engine**, không phải một
đống tài liệu chung chung. Bài xuất hiện ở nhiều truy vấn được gộp `query_tags` — chỉ
số hữu ích: bài phục vụ nhiều cơ chế thường là tổng quan tốt, đáng đọc kỹ.

### Dogfooding

Sau khi RAG chạy, dùng chính nó để dựng và kiểm chứng danh mục tài liệu tham khảo của
khóa luận. Vừa tiện vừa là một minh chứng đẹp trong báo cáo.

---

## 5. Chunking

**Chiến lược: theo cấu trúc trước, theo độ dài sau.**

Vì sao không cắt cứng theo số token: một câu khẳng định cơ chế —
*"PBLH below 500 m was associated with a 2.3-fold increase in PM2.5"* — bị cắt đôi
thì đoạn nào cũng mất nghĩa, và **trích dẫn sẽ không chứng minh được điều nó nói**.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Nhận diện mục | regex Abstract/Introduction/Methods/Results/Discussion/Conclusion | không nhận ra → một mục `body` |
| Kích thước | 320 từ ≈ 400–450 token | |
| Chồng lấn | 15%, tính theo **câu nguyên vẹn** | một ý trải qua ranh giới vẫn còn ngữ cảnh |
| Ranh giới | **không bao giờ cắt giữa câu** | có unit test |
| Tiêu đề bài | ghép vào đầu mỗi chunk | embedding của một đoạn Methods rời rạc rất dễ lạc chủ đề |

**Với bài chỉ có abstract** (đa số, do ràng buộc bản quyền): một abstract thường vừa
gọn trong 1–2 chunk. Đây không phải hạn chế nghiêm trọng như thoạt nghe — abstract
vốn là phần **đậm đặc kết luận nhất** của bài báo.

---

## 6. Embedding

| Backend | Model | Dùng khi nào |
|---|---|---|
| `hf` | `BAAI/bge-m3` | **Mọi kết quả trong khóa luận.** Đa ngôn ngữ → xử lý được cả tài liệu tiếng Việt |
| `hash` | — | **Chỉ unit test.** Băm n-gram, offline, không cần torch |

### Vì sao vẫn giữ backend `hash`

Nếu bắt buộc phải có torch (~2 GB) mới chạy được test, thì test sẽ không chạy trên CI
và trên máy yếu — và trong thực tế là **không ai chạy**. Backend `hash` giữ cho toàn bộ
đường ống RAG luôn được kiểm thử.

Nó **không có chất lượng ngữ nghĩa**: "boundary layer" và "mixing height" sẽ không gần
nhau. `hash_backend_warning()` nhắc điều đó ở mọi nơi cần thiết, và
`scripts/index_corpus.py` in cảnh báo khi phát hiện.

---

## 7. Rerank

Vì sao cần thêm một bước sau vector search: **bi-encoder nén cả đoạn văn vào một
vector trước khi biết truy vấn là gì**. Nó giỏi gọi ra ~20 ứng viên nhưng xếp hạng
chưa sắc. Cross-encoder đọc *đồng thời* truy vấn và đoạn văn, nên phân biệt được:

- "bài này **nhắc tới** boundary layer"
- "bài này **chứng minh** boundary layer thấp làm tăng PM2.5"

Trong hệ thống này **precision quan trọng hơn recall**: một trích dẫn sai làm hỏng
tính "evidence-grounded" nặng hơn là thiếu một trích dẫn đúng.

**Chi tiết dễ bỏ sót**: cross-encoder trả logit, cosine trả [−1, 1]. `_to_unit()` ép
logit qua sigmoid về [0, 1] để **cùng thang** — nếu không, bật/tắt reranker sẽ vô tình
đổi luôn độ chặt của cổng chặn và làm hỏng so sánh ablation.

---

## 8. Vector store

**Qdrant** là backend chính thức (`docker compose up -d qdrant`, UI tại
`localhost:6333/dashboard`). Distance = COSINE vì mọi embedder đều trả vector đã
chuẩn hóa L2.

`InMemoryStore` **chỉ dùng cho test**. Nếu Qdrant không kết nối được, `QdrantStore`
**ném lỗi rõ ràng** chứ không âm thầm chuyển sang bộ nhớ — im lặng thất bại ở tầng
này sẽ tạo ra câu trả lời không có trích dẫn mà không ai biết.

`Chunk.point_id()` sinh UUID xác định từ `chunk_id`, nên index lại cùng một chunk sẽ
**ghi đè** thay vì nhân bản.

---

## 9. Tính `rag_support`

```
support = 0.65 · độ_khớp_tốt_nhất_chuẩn_hóa  +  0.35 · độ_phủ

    độ_khớp = (best_score − min_score) / (1 − min_score)
    độ_phủ  = số_trích_dẫn_giữ_lại / top_k
```

Hai thành phần vì hai thứ khác nhau đều đáng kể:
- Một tài liệu khớp **rất sát** là bằng chứng mạnh.
- **Nhiều tài liệu độc lập** cùng nói một điều là *đồng thuận khoa học* — thứ mà một
  bài đơn lẻ không thay thế được.

---

## 10. Cấp nhãn bằng chứng

Nhãn `E1, E2, ...` được cấp phát **toàn cục** qua tất cả các cơ chế trong một lần trả
lời. Cùng một đoạn văn chống lưng cho hai cơ chế thì mang **cùng một nhãn**.

Nếu không: câu trả lời sẽ có `[E2]` và `[E5]` trỏ về cùng một chỗ, và người đọc tưởng
là hai bằng chứng độc lập — một dạng thổi phồng bằng chứng rất khó phát hiện.
Có unit test: `test_evidence_ids_are_shared_across_mechanisms`.

---

## 11. Tham số cấu hình được (đều là biến ablation tiềm năng)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `fetch_k` | 20 | Ứng viên lấy trước rerank |
| `top_k` | 3 | Trích dẫn giữ lại mỗi cơ chế — giữ ít để câu trả lời không loãng |
| `min_score` | 0.30 | **Cổng chặn.** Cần hiệu chỉnh trên bộ query có nhãn (docs/06 §3), không đoán |
| `chunk_words` | 320 | |
| `overlap_ratio` | 0.15 | |

---

## 12. Quy trình vận hành

```powershell
# 1. Bật Qdrant
docker compose up -d qdrant

# 2. Cài phụ thuộc
pip install -r requirements-embed.txt      # gồm cả requirements-rag.txt

# 3. Tải corpus (chỉ metadata + abstract)
python scripts/build_corpus.py --per-query 20

# 4. Chunk + embed + nạp
python scripts/index_corpus.py --embedding hf --recreate

# 5. Kiểm tra
# http://localhost:6333/dashboard
```

Sau đó truyền `GatedRetriever` vào `ExplanationPipeline(retriever=...)` để bật tầng
RAG trong pipeline (hiện `scripts/demo_explain.py` để `retriever=None`).
