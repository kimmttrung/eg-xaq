# ==============================================================================
# EG-XAQ — Embed corpus bằng GPU trên Kaggle rồi nạp vào Qdrant Cloud
# ==============================================================================
#
# Dán từng khối `# %%` vào một cell của Kaggle Notebook.
#
# CHUẨN BỊ TRƯỚC KHI CHẠY
# -----------------------
# 1. Local:  python scripts/export_for_kaggle.py
#            → data/corpus/kaggle/{chunks.jsonl, queries.json}
#
# 2. Kaggle: upload thư mục đó thành Dataset (đặt tên: egxaq-corpus)
#
# 3. Kaggle: Settings → Accelerator = GPU T4 x2
#            Settings → Internet = ON   (cần xác minh số điện thoại một lần)
#
# 4. Kaggle: Add-ons → Secrets, thêm hai secret:
#              QDRANT_URL       https://xxxx.aws.cloud.qdrant.io:6333
#              QDRANT_API_KEY   <key của cluster>
#            TUYỆT ĐỐI không viết key thẳng vào notebook — notebook có thể public.
#
# NGUYÊN TẮC BẤT BIẾN
# -------------------
# Notebook này KHÔNG cắt chunk, KHÔNG dựng payload, KHÔNG quyết định point_id.
# Tất cả đã được repo quyết định và ghi sẵn trong chunks.jsonl. Việc duy nhất ở
# đây là biến `text` thành vector. Giữ đúng ranh giới này thì logic không bao giờ
# bị nhân đôi giữa repo và notebook.


# %%
# ------------------------------------------------------------------ cài đặt
# Kaggle đã có sẵn torch. Chỉ cần thêm hai gói.
!pip install -q sentence-transformers qdrant-client


# %%
# ------------------------------------------------------------------ cấu hình
import json
from pathlib import Path

# ⚠ Đổi cho khớp tên Dataset bạn upload.
DATA_DIR = Path("/kaggle/input/egxaq-corpus")
OUT_DIR = Path("/kaggle/working")

MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "egxaq_papers"
BATCH_SIZE = 32          # T4 16GB dư sức; giảm xuống nếu gặp OOM
RECREATE_COLLECTION = True   # True = xóa collection cũ rồi tạo lại

chunks = [json.loads(line) for line in (DATA_DIR / "chunks.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip()]
query_spec = json.loads((DATA_DIR / "queries.json").read_text(encoding="utf-8"))
queries = query_spec["queries"]

print(f"Chunk   : {len(chunks)}")
print(f"Truy vấn: {len(queries)}")
print(f"Model   : {MODEL_NAME} (repo mong đợi {query_spec['model']}, dim {query_spec['dim']})")
assert MODEL_NAME == query_spec["model"], (
    "Model ở đây phải TRÙNG với EGXAQ_EMBEDDING_MODEL trong .env. "
    "Embed tài liệu bằng model này mà truy vấn bằng model khác thì kết quả là rác."
)


# %%
# ------------------------------------------------------------------ embed
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("⚠ Chưa bật GPU. Settings → Accelerator → GPU T4.")

model = SentenceTransformer(MODEL_NAME, device=device)
dim = model.get_sentence_embedding_dimension()
print(f"Số chiều thật của model: {dim}")
assert dim == query_spec["dim"], (
    f"Model trả {dim} chiều nhưng .env khai {query_spec['dim']}. "
    "Sửa EGXAQ_EMBEDDING_DIM cho khớp, nếu không Qdrant sẽ từ chối."
)

# normalize_embeddings=True là BẮT BUỘC: collection dùng distance COSINE và
# repo giả định vector đã chuẩn hóa L2 ở mọi nơi.
doc_vectors = model.encode(
    [c["text"] for c in chunks],
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print("Vector tài liệu:", doc_vectors.shape)

query_vectors = model.encode(
    queries,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True,
)
print("Vector truy vấn:", query_vectors.shape)


# %%
# ------------------------------------------------------------------ nạp Qdrant
from kaggle_secrets import UserSecretsClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

secrets = UserSecretsClient()
client = QdrantClient(
    url=secrets.get_secret("QDRANT_URL"),
    api_key=secrets.get_secret("QDRANT_API_KEY"),
    timeout=120,
)

if RECREATE_COLLECTION and client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )

# point_id lấy thẳng từ repo (UUID suy ra từ chunk_id) nên index lại sẽ GHI ĐÈ,
# không nhân bản điểm.
UPLOAD_BATCH = 128
for start in range(0, len(chunks), UPLOAD_BATCH):
    window = chunks[start : start + UPLOAD_BATCH]
    client.upsert(
        collection_name=COLLECTION,
        points=[
            qmodels.PointStruct(
                id=item["point_id"],
                vector=vector.tolist(),
                payload=item["payload"],
            )
            for item, vector in zip(window, doc_vectors[start : start + UPLOAD_BATCH])
        ],
        wait=True,
    )
    print(f"  đã nạp {min(start + UPLOAD_BATCH, len(chunks))}/{len(chunks)}", end="\r")

print(f"\nCollection '{COLLECTION}': {client.count(COLLECTION, exact=True).count} điểm.")


# %%
# ------------------------------------------------------------------ xuất truy vấn
# File này mang về máy → data/corpus/query_vectors.json.
# Nhờ nó máy local KHÔNG cần torch: tra bảng thay vì chạy model.
out_path = OUT_DIR / "query_vectors.json"
out_path.write_text(
    json.dumps(
        {
            "model": MODEL_NAME,
            "dim": int(dim),
            "vectors": {q: v.tolist() for q, v in zip(queries, query_vectors)},
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"Đã ghi {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
print("Tải file này từ tab Output của Kaggle về data/corpus/query_vectors.json")


# %%
# ------------------------------------------------------------------ kiểm chứng
# Thử truy xuất ngay trên Kaggle để biết corpus có thực sự chống lưng được cơ chế
# nào không — rẻ hơn nhiều so với phát hiện ra lúc đã về máy.
for q, qv in zip(queries, query_vectors):
    hits = client.query_points(
        collection_name=COLLECTION, query=qv.tolist(), limit=3, with_payload=True
    ).points
    best = hits[0].score if hits else 0.0
    print(f"\n[{best:.3f}] {q[:72]}")
    for h in hits[:2]:
        print(f"    {h.score:.3f}  {(h.payload.get('title') or '')[:66]}")

print(
    "\nDùng phân bố điểm ở trên để chọn RetrievalConfig.min_score — "
    "cổng 0.30 là giá trị cho backend 'hash', KHÔNG mang sang BGE-M3 được."
)
