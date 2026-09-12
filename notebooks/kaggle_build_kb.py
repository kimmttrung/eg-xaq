# ==============================================================================
# EG-XAQ — Dựng toàn bộ Knowledge Base trên Kaggle (end-to-end)
# ==============================================================================
#
# Một notebook làm trọn: tải corpus → cắt chunk → dựng payload → embed BGE-M3
# trên GPU → nạp Qdrant Cloud → xuất vector truy vấn cho máy local.
#
# ------------------------------------------------------------------------------
# NGUYÊN TẮC: NOTEBOOK KHÔNG TỰ VIẾT LẠI LOGIC
# ------------------------------------------------------------------------------
# Notebook clone repo rồi GỌI CHÍNH HÀM CỦA REPO (build_corpus, chunk_paper,
# Chunk.point_id, Chunk.payload). Không chép lại cách cắt chunk, không tự chế
# payload, không tự sinh id.
#
# Lý do: `chunk_id` → `point_id` → payload là hợp đồng giữa lúc index và lúc truy
# xuất. Nếu notebook cắt theo một kiểu còn repo cắt theo kiểu khác, Qdrant vẫn nạp
# thành công, truy xuất vẫn trả kết quả, và không có lỗi nào hiện ra — chỉ là
# trích dẫn trỏ sai chỗ. Đúng loại hỏng-mà-không-giống-hỏng mà dự án này rất kỵ.
#
# ------------------------------------------------------------------------------
# CHUẨN BỊ TRÊN KAGGLE (làm một lần)
# ------------------------------------------------------------------------------
# Settings → Accelerator          : GPU T4 x2
# Settings → Internet             : ON  (cần xác minh số điện thoại một lần)
# Add-ons → Secrets               : thêm QDRANT_URL và QDRANT_API_KEY
#
# Chạy tự động: Save Version → "Save & Run All (Commit)".
# Nhớ bật Internet + GPU + đính Secrets TRƯỚC khi commit, nếu không bản chạy nền
# sẽ đứt giữa chừng.
#
# ------------------------------------------------------------------------------
# SAU KHI CHẠY XONG — lấy ở tab Output
# ------------------------------------------------------------------------------
#   query_vectors.json   → chép vào data/corpus/  (thay thế torch ở local)
#   papers.jsonl         → chép vào data/corpus/  (để chạy --rag memory offline)
#
# Rồi đặt trong .env ở máy:
#   EGXAQ_EMBEDDING_BACKEND=precomputed
#   EGXAQ_QDRANT_URL=<url cluster>
#   EGXAQ_QDRANT_API_KEY=<api key>


# %%
# ------------------------------------------------------------------ 1. cài đặt
# Kaggle có sẵn torch, numpy, requests, pyyaml. Chỉ thiếu ba gói này.
!pip install -q sentence-transformers qdrant-client pydantic-settings


# %%
# ------------------------------------------------------------------ 2. lấy code
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/kimmttrung/eg-xaq.git"
REPO_DIR = Path("/kaggle/working/eg-xaq")

if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

sys.path.insert(0, str(REPO_DIR / "src"))

# Cấu hình đọc từ biến môi trường (repo clone về không có .env — .env nằm trong
# .gitignore, và đó là điều đúng đắn).
os.environ["OPENALEX_MAILTO"] = "mttrung231@gmail.com"   # ⚠ đổi thành email của bạn
os.environ["EGXAQ_EMBEDDING_MODEL"] = "BAAI/bge-m3"
os.environ["EGXAQ_EMBEDDING_DIM"] = "1024"
os.environ["EGXAQ_QDRANT_COLLECTION"] = "egxaq_papers"

print(subprocess.run(["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"],
                     capture_output=True, text=True).stdout)


# %%
# ------------------------------------------------------------------ 3. tham số
# Tăng PER_QUERY và SNOWBALL_SEEDS để corpus lớn hơn. Mốc tham khảo:
#   per_query=8,  seeds=8   →  ~400 bài
#   per_query=30, seeds=30  →  ~1500–3000 bài
PER_QUERY = 30
YEAR_MIN = 2015
USE_SNOWBALL = True
SNOWBALL_SEEDS = 30
SNOWBALL_MIN_CITATIONS = 5

CHUNK_WORDS = 320
EMBED_BATCH = 32          # T4 16GB thoải mái; giảm nếu OOM
UPLOAD_BATCH = 128
RECREATE_COLLECTION = True

OUT_DIR = Path("/kaggle/working")


# %%
# ------------------------------------------------------------------ 4. corpus
from kb import get_knowledge_base
from rag.corpus import SEED_QUERIES, build_corpus, corpus_stats, licensing_report, save_corpus, snowball

kb = get_knowledge_base()
mechanism_queries = [
    " ".join(m.rag_query.split()) for m in kb.mechanisms.values() if m.rag_query.strip()
]
queries = list(dict.fromkeys(mechanism_queries + SEED_QUERIES))
print(f"Truy vấn: {len(queries)} ({len(mechanism_queries)} từ cơ chế + {len(SEED_QUERIES)} nền)\n")

papers = build_corpus(
    queries, per_query=PER_QUERY, year_min=YEAR_MIN, with_references=USE_SNOWBALL
)

if USE_SNOWBALL:
    print()
    seeds = sorted(papers, key=lambda p: p.cited_by_count, reverse=True)
    found = snowball(
        seeds, year_min=YEAR_MIN,
        min_citations=SNOWBALL_MIN_CITATIONS, max_seeds=SNOWBALL_SEEDS,
    )
    known = {(p.doi or p.paper_id).lower() for p in papers}
    added = [p for p in found if (p.doi or p.paper_id).lower() not in known]
    papers.extend(added)
    print(f"  Snowball bổ sung {len(added)} bài (tổng {len(papers)}).")

save_corpus(papers, OUT_DIR / "papers.jsonl")
print()
for key, value in corpus_stats(papers).items():
    print(f"  {key:<20} {value}")

print("\n" + "=" * 60)
print("HỒ SƠ BẢN QUYỀN (đưa vào phụ lục khóa luận)")
print("=" * 60)
print(licensing_report(papers))


# %%
# ------------------------------------------------------------------ 5. chunk
# Dùng chunk_paper CỦA REPO — quyết định chunk_id, point_id và payload.
from rag.chunking import chunk_paper

chunks = [c for p in papers for c in chunk_paper(p, chunk_words=CHUNK_WORDS)]
print(f"{len(papers)} bài → {len(chunks)} chunk")
print(f"\nVí dụ payload sẽ nằm trong Qdrant:")
for key, value in chunks[0].payload().items():
    shown = str(value)
    print(f"  {key:<16} {shown[:70]}{'…' if len(shown) > 70 else ''}")


# %%
# ------------------------------------------------------------------ 6. embed
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cpu":
    print("⚠ Chưa bật GPU: Settings → Accelerator → GPU T4.")

model = SentenceTransformer(os.environ["EGXAQ_EMBEDDING_MODEL"], device=device)
dim = model.get_sentence_embedding_dimension()
assert dim == int(os.environ["EGXAQ_EMBEDDING_DIM"]), (
    f"Model trả {dim} chiều nhưng cấu hình khai {os.environ['EGXAQ_EMBEDDING_DIM']}. "
    "Sửa EGXAQ_EMBEDDING_DIM ở cả đây lẫn .env, nếu không Qdrant sẽ từ chối."
)

# normalize_embeddings=True là BẮT BUỘC: collection dùng COSINE và repo giả định
# vector đã chuẩn hóa L2 ở mọi nơi.
doc_vectors = model.encode(
    [c.text for c in chunks],
    batch_size=EMBED_BATCH, normalize_embeddings=True,
    show_progress_bar=True, convert_to_numpy=True,
)
print("Vector tài liệu:", doc_vectors.shape)

# 12 truy vấn của cơ chế — embed CÙNG model, CÙNG lượt. Đây là thứ thay thế torch
# ở máy local: PrecomputedQueryEmbedder chỉ tra bảng này.
from rag.embedding import normalize_query

mech_queries = sorted({normalize_query(q) for q in mechanism_queries})
query_vectors = model.encode(
    mech_queries, batch_size=EMBED_BATCH, normalize_embeddings=True, convert_to_numpy=True
)
print("Vector truy vấn:", query_vectors.shape)


# %%
# ------------------------------------------------------------------ 7. Qdrant
from kaggle_secrets import UserSecretsClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

secrets = UserSecretsClient()
client = QdrantClient(
    url=secrets.get_secret("QDRANT_URL"),
    api_key=secrets.get_secret("QDRANT_API_KEY"),
    timeout=120,
)
COLLECTION = os.environ["EGXAQ_QDRANT_COLLECTION"]

if RECREATE_COLLECTION and client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )

# point_id suy ra từ chunk_id nên index lại sẽ GHI ĐÈ, không nhân bản điểm.
for start in range(0, len(chunks), UPLOAD_BATCH):
    window = chunks[start : start + UPLOAD_BATCH]
    client.upsert(
        collection_name=COLLECTION,
        points=[
            qmodels.PointStruct(id=c.point_id(), vector=v.tolist(), payload=c.payload())
            for c, v in zip(window, doc_vectors[start : start + UPLOAD_BATCH])
        ],
        wait=True,
    )
    print(f"  đã nạp {min(start + UPLOAD_BATCH, len(chunks))}/{len(chunks)}", end="\r")

print(f"\nCollection '{COLLECTION}': {client.count(COLLECTION, exact=True).count} điểm.")


# %%
# ------------------------------------------------------------------ 8. xuất
import json

out_path = OUT_DIR / "query_vectors.json"
out_path.write_text(
    json.dumps(
        {
            "model": os.environ["EGXAQ_EMBEDDING_MODEL"],
            "dim": int(dim),
            "vectors": {q: v.tolist() for q, v in zip(mech_queries, query_vectors)},
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"{out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)  ← chép vào data/corpus/")
print(f"papers.jsonl  ({(OUT_DIR / 'papers.jsonl').stat().st_size / 1024:.0f} KB)  ← chép vào data/corpus/")


# %%
# ------------------------------------------------------------------ 9. hiệu chỉnh cổng
# Phân bố điểm dưới đây là CĂN CỨ để chọn RetrievalConfig.min_score.
# Cổng 0.30 hiện trong repo là giá trị của backend `hash` — KHÔNG mang sang BGE-M3.
import numpy as np

best_scores = []
for q, qv in zip(mech_queries, query_vectors):
    hits = client.query_points(
        collection_name=COLLECTION, query=qv.tolist(), limit=3, with_payload=True
    ).points
    best = hits[0].score if hits else 0.0
    best_scores.append(best)
    print(f"\n[{best:.3f}] {q[:70]}")
    for h in hits[:2]:
        print(f"    {h.score:.3f}  {(h.payload.get('title') or '')[:64]}")

arr = np.array(best_scores)
print("\n" + "=" * 60)
print(f"Điểm khớp tốt nhất qua {len(arr)} cơ chế:")
print(f"  min={arr.min():.3f}  p25={np.percentile(arr, 25):.3f}  "
      f"trung vị={np.median(arr):.3f}  max={arr.max():.3f}")
print(
    "\nChọn min_score sao cho các cơ chế thực sự có tài liệu thì vượt cổng, còn cơ\n"
    "chế không có thì bị chặn. ĐỌC KỸ tiêu đề in ở trên trước khi chốt: điểm cao\n"
    "mà tiêu đề lạc đề nghĩa là cổng đang cho rác lọt, không phải corpus tốt."
)

# Dọn repo khỏi Output cho nhẹ — để cuối cùng, sau khi mọi cell đã dùng xong.
import shutil

shutil.rmtree(REPO_DIR, ignore_errors=True)
