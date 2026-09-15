"""Đo độ phủ ĐỊA LÝ của corpus — corpus nói về đâu, và có nói về Việt Nam không.

    python scripts/corpus_coverage.py
    python scripts/corpus_coverage.py --by-mechanism

VÌ SAO CẦN ĐO CHỨ KHÔNG ĐOÁN
============================

Corpus xây bằng truy vấn tiếng Anh nên gần như chắc chắn lệch về nơi có nhiều
công bố quốc tế: Trung Quốc, Ấn Độ, châu Âu, Bắc Mỹ. Việt Nam có rất ít bài trên
các tạp chí mà OpenAlex phủ tốt.

Điều đó KHÔNG làm corpus vô dụng — xem giải thích bên dưới — nhưng nó là một hạn
chế phải nêu thành con số trong khóa luận, không phải giấu đi. Hội đồng hỏi
"tài liệu của em nói về Trung Quốc, sao áp cho Hà Nội được?" thì câu trả lời
phải là một bảng, không phải một lời trấn an.

TẦNG RAG CHỨNG MINH ĐIỀU GÌ
---------------------------
Trích dẫn gắn vào CƠ CHẾ, không gắn vào địa điểm. Mệnh đề được chống lưng là
"lớp xáo trộn nông làm PM2.5 tích tụ" — vật lý khí quyển, không phụ thuộc biên
giới. Lớp xáo trộn thấp giữ bụi ở Bắc Kinh, Hà Nội, Delhi hay Kraków theo cùng
một cơ chế.

CÁI GÌ THÌ PHỤ THUỘC ĐỊA PHƯƠNG
-------------------------------
1. NGƯỠNG   — PBLH < 500 m rút từ văn liệu haze Đông Á. Khí hậu gió mùa nhiệt
              đới của Hà Nội có thể cần ngưỡng khác. Đây là việc của trường
              `calibration` trong rules.yaml, cần dữ liệu GFS của lab.
2. NGUỒN    — haze mùa đông Trung Quốc là than sưởi + công nghiệp nặng; Hà Nội là
              giao thông + đốt rơm rạ + lò gạch + bụi xây dựng.
3. TRỌNG SỐ — cơ chế nào chiếm ưu thế ở đâu là khác nhau.
4. KHÍ HẬU  — nồm ẩm là hiện tượng riêng của đồng bằng Bắc Bộ.

Nên đọc kết quả script này như sau: tỉ lệ Đông Á cao là BÌNH THƯỜNG và chấp nhận
được cho việc chứng minh cơ chế; tỉ lệ Việt Nam/ĐNÁ thấp là hạn chế cần nêu rõ khi
phát biểu bất cứ điều gì mang tính địa phương.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from config import get_settings
from kb import get_knowledge_base
from rag.corpus import load_corpus
from rag.models import Paper

#: Vùng → từ khóa nhận dạng, tìm trong tiêu đề + abstract + tên tạp chí.
#: Xét theo THỨ TỰ này: bài nhắc cả Việt Nam lẫn Trung Quốc được tính cho Việt Nam,
#: vì cái ta quan tâm là "có liên quan tới địa bàn nghiên cứu không".
REGIONS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Việt Nam",
        (
            "vietnam",
            "viet nam",
            "vietnamese",
            "hanoi",
            "ha noi",
            "red river",
            "ho chi minh",
            "da nang",
            "mekong",
            "indochina",
        ),
    ),
    (
        "Đông Nam Á khác",
        (
            "southeast asia",
            "south-east asia",
            "thailand",
            "bangkok",
            "indonesia",
            "jakarta",
            "malaysia",
            "singapore",
            "philippines",
            "manila",
            "myanmar",
            "laos",
            "cambodia",
            "borneo",
            "sumatra",
        ),
    ),
    (
        "Đông Á",
        (
            "china",
            "chinese",
            "beijing",
            "shanghai",
            "guangzhou",
            "pearl river",
            "yangtze",
            "sichuan",
            "north china plain",
            "korea",
            "seoul",
            "japan",
            "tokyo",
            "taiwan",
            "hong kong",
        ),
    ),
    (
        "Nam Á",
        (
            "india",
            "indian",
            "delhi",
            "indo-gangetic",
            "pakistan",
            "bangladesh",
            "dhaka",
            "nepal",
            "kathmandu",
            "himalaya",
        ),
    ),
    (
        "Châu Âu",
        (
            "europe",
            "european",
            "po valley",
            "united kingdom",
            "london",
            "germany",
            "poland",
            "krakow",
            "netherlands",
            "france",
            "paris",
            "spain",
            "italy",
        ),
    ),
    (
        "Bắc Mỹ",
        ("united states", "u.s.", " usa ", "california", "canada", "mexico city", "north america"),
    ),
]


def region_of(paper: Paper) -> str:
    """Vùng địa lý mà bài báo nói tới. Không nhận ra → 'Không rõ / tổng quát'."""
    haystack = " ".join(
        [paper.title or "", paper.abstract or "", paper.venue or "", " ".join(paper.topics)]
    ).lower()
    for name, keywords in REGIONS:
        if any(keyword in haystack for keyword in keywords):
            return name
    return "Không rõ / tổng quát"


def main() -> int:
    parser = argparse.ArgumentParser(description="Đo độ phủ địa lý của corpus")
    parser.add_argument("--corpus", default=None, help="Mặc định data/corpus/papers.jsonl")
    parser.add_argument(
        "--by-mechanism", action="store_true", help="Tách theo từng cơ chế (dùng query_tags)"
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus_path = Path(args.corpus) if args.corpus else settings.corpus_dir / "papers.jsonl"
    papers = list(load_corpus(corpus_path))

    tagged = [(paper, region_of(paper)) for paper in papers]
    counts = Counter(region for _, region in tagged)
    total = len(papers)

    print(f"Corpus: {total} bài từ {corpus_path}\n")
    print("=" * 62)
    print("ĐỘ PHỦ ĐỊA LÝ")
    print("=" * 62)
    for region, count in counts.most_common():
        bar = "█" * round(40 * count / total)
        print(f"  {region:<22} {count:>4}  {count / total:>5.0%}  {bar}")

    local = counts["Việt Nam"] + counts["Đông Nam Á khác"]
    print()
    print(f"  Việt Nam + ĐNÁ : {local}/{total} ({local / total:.0%})")
    if counts["Việt Nam"] == 0:
        print("  ⚠ KHÔNG có bài nào nhắc tới Việt Nam.")

    if args.by_mechanism:
        _print_by_mechanism(tagged)

    _print_verdict(counts, total)
    return 0


def _print_by_mechanism(tagged: list[tuple[Paper, str]]) -> None:
    """Tách theo cơ chế nhờ `query_tags` — bài nào được truy vấn nào tìm ra.

    Đây mới là con số dùng được: một cơ chế mà TOÀN BỘ tài liệu chống lưng đều
    đến từ một vùng khí hậu khác là chỗ cần thận trọng nhất khi phát biểu.
    """
    kb = get_knowledge_base()
    query_to_mech = {
        " ".join(m.rag_query.split()): m.id for m in kb.mechanisms.values() if m.rag_query.strip()
    }

    by_mech: dict[str, Counter] = defaultdict(Counter)
    for paper, region in tagged:
        for tag in paper.query_tags:
            mech_id = query_to_mech.get(" ".join(tag.split()))
            if mech_id:
                by_mech[mech_id][region] += 1

    print("\n" + "=" * 62)
    print("THEO TỪNG CƠ CHẾ")
    print("=" * 62)
    for mech_id in sorted(by_mech):
        counter = by_mech[mech_id]
        n = sum(counter.values())
        local = counter["Việt Nam"] + counter["Đông Nam Á khác"]
        flag = "  ⚠ không có tài liệu khu vực" if local == 0 else ""
        print(f"\n  {mech_id}  ({n} bài, {local} ĐNÁ){flag}")
        for region, count in counter.most_common(3):
            print(f"      {region:<22} {count}")


def _print_verdict(counts: Counter, total: int) -> None:
    local_share = (counts["Việt Nam"] + counts["Đông Nam Á khác"]) / total if total else 0.0

    print("\n" + "=" * 62)
    print("ĐỌC KẾT QUẢ NÀY THẾ NÀO")
    print("=" * 62)
    print(
        "Corpus lệch về Đông Á là chuyện BÌNH THƯỜNG và không làm hỏng tầng RAG:\n"
        "trích dẫn gắn vào CƠ CHẾ, mà cơ chế khí quyển thì không theo biên giới.\n"
        "Lớp xáo trộn nông giữ bụi ở Bắc Kinh và Hà Nội theo cùng một vật lý.\n"
    )
    if local_share < 0.15:
        print(
            f"⚠ Nhưng chỉ {local_share:.0%} tài liệu thuộc Việt Nam/ĐNÁ. Hệ quả BẮT BUỘC\n"
            "  phải nêu trong khóa luận:\n"
            "    • Ngưỡng trong rules.yaml là ngưỡng VĂN LIỆU ĐÔNG Á, chưa hiệu chỉnh\n"
            "      cho khí hậu gió mùa nhiệt đới → cần dữ liệu GFS của lab.\n"
            "    • Cơ cấu nguồn thải Hà Nội (đốt rơm rạ, lò gạch, giao thông) khác hẳn\n"
            "      haze than sưởi Trung Quốc → không suy diễn tỉ lệ đóng góp.\n"
            "    • Mọi phát biểu mang tính ĐỊA PHƯƠNG phải hạ độ tin cậy tương ứng.\n"
        )
        print(
            "  Cách cải thiện: thêm truy vấn địa phương vào SEED_QUERIES\n"
            "  (rag/corpus.py) rồi chạy lại notebook Kaggle."
        )


if __name__ == "__main__":
    raise SystemExit(main())
