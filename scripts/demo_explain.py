"""Demo end-to-end: câu hỏi → chuỗi bằng chứng → câu trả lời.

Chạy được ngay, không cần data lab, không cần API key, không cần Docker.

    python scripts/demo_explain.py --episode winter_inversion
    python scripts/demo_explain.py --episode biomass_burning --ablation C
    python scripts/demo_explain.py --episode winter_inversion --xai-mode conflicting
    python scripts/demo_explain.py --list
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401  (phải import trước mọi module của dự án)
from data.mock import EPISODES, HANOI_LAT, HANOI_LON, MockObservationProvider
from narrator.narrator import DryRunNarrator, get_narrator
from pipeline import ExplanationPipeline, PipelineConfig
from reasoning.kg import to_mermaid
from xai.mock import MockAttributionProvider

DEFAULT_QUESTIONS = {
    "winter_inversion": "Vì sao hôm nay Hà Nội ô nhiễm nặng dù không có nguồn thải bất thường?",
    "biomass_burning": "Vì sao chất lượng không khí Hà Nội xấu đột ngột hôm nay?",
    "cold_surge_clean": "Vì sao hôm nay không khí Hà Nội lại trong lành?",
    "rain_washout": "Mưa hôm nay có làm sạch không khí không?",
    "humid_stagnant": "Trời nồm ẩm thế này thì bụi mịn ra sao?",
    "sparse_data": "Vì sao hôm nay Hà Nội ô nhiễm?",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo pipeline EG-XAQ")
    parser.add_argument("--episode", default="winter_inversion", choices=sorted(EPISODES))
    parser.add_argument("--question", default=None, help="Ghi đè câu hỏi mặc định")
    parser.add_argument(
        "--ablation", default="E", choices=list("ABCDE"), help="Cấu hình ablation (mặc định E)"
    )
    parser.add_argument(
        "--xai-mode",
        default="aligned",
        choices=["aligned", "conflicting", "spurious"],
        help="Chế độ SHAP giả lập — 'conflicting' để xem cơ chế phát hiện mâu thuẫn",
    )
    parser.add_argument("--step", type=int, default=0, help="Bước dự báo t+step (0..9)")
    parser.add_argument("--narrator", default="dryrun", choices=["dryrun", "anthropic"])
    parser.add_argument("--json", action="store_true", help="In EvidenceBundle dạng JSON")
    parser.add_argument("--mermaid", action="store_true", help="In sơ đồ KG của lần trả lời này")
    parser.add_argument("--list", action="store_true", help="Liệt kê kịch bản rồi thoát")
    args = parser.parse_args()

    if args.list:
        print("Các kịch bản có sẵn:\n")
        for name, spec in EPISODES.items():
            print(f"  {name:<20} {spec.get('_note', '')}")
        return 0

    question = args.question or DEFAULT_QUESTIONS.get(args.episode, "Vì sao không khí hôm nay xấu?")
    observations = MockObservationProvider(episode=args.episode)
    date = MockObservationProvider.default_date(args.episode)

    pipeline = ExplanationPipeline(
        observation_provider=observations,
        attribution_provider=MockAttributionProvider(mode=args.xai_mode),
        retriever=None,  # bật RAG bằng scripts/index_corpus.py rồi truyền GatedRetriever
        narrator=get_narrator(args.narrator) if args.narrator != "dryrun" else DryRunNarrator(),
        config=PipelineConfig.ablation(args.ablation),
    )

    print("=" * 78)
    print(f"KỊCH BẢN   : {args.episode} — {MockObservationProvider.describe(args.episode)}")
    print(f"CÂU HỎI    : {question}")
    print(f"CẤU HÌNH   : ablation {args.ablation} | SHAP {args.xai_mode} | t+{args.step}")
    print("⚠ DỮ LIỆU GIẢ LẬP — không dùng cho kết quả khóa luận.")
    print("=" * 78)

    bundle = pipeline.build_bundle(
        question, lat=HANOI_LAT, lon=HANOI_LON, date=date, step=args.step
    )

    if args.json:
        print(bundle.model_dump_json(indent=2))
        return 0

    answer = pipeline.narrator.narrate(bundle)
    print()
    print(answer.text)

    if args.mermaid:
        print("\n" + "=" * 78)
        print("SƠ ĐỒ KNOWLEDGE GRAPH (dán vào khóa luận)")
        print("=" * 78)
        ids = [h.mechanism_id for h in [*bundle.hypotheses, *bundle.suppressors]]
        print("```mermaid")
        print(to_mermaid(pipeline.graph, ids))
        print("```")

    print("\n" + "-" * 78)
    print(
        json.dumps(
            {
                "hypotheses": len(bundle.hypotheses),
                "suppressors": len(bundle.suppressors),
                "conflicts": len(bundle.conflicts),
                "confidence": bundle.overall_confidence.value,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
