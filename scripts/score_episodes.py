"""Chấm điểm hệ thống trên bộ episode đã gán nhãn — ra bảng kết quả chính.

    python scripts/score_episodes.py --smoke                 # kiểm thử bộ chấm
    python scripts/score_episodes.py                         # data/eval/episodes.yaml
    python scripts/score_episodes.py --rag qdrant --ablations D,E

Mỗi episode chạy qua từng cấu hình ablation (docs/06 §4), so với nhãn, rồi gộp:
F1 chặt / nới, top-1, tỉ lệ khẳng định cơ chế bị loại trừ, và độ đúng loại câu trả lời.

EPISODE THẬT KHÔNG BAO GIỜ CHẠY TRÊN DỮ LIỆU GIẢ
================================================
Nếu `EGXAQ_DATA_BACKEND=mock`, episode thật bị BỎ QUA và liệt kê rõ — không âm thầm
lấy dữ liệu mock của một kịch bản bất kỳ. Số liệu sinh từ dữ liệu giả cho một ngày
thật trông y hệt số liệu thật, và đó chính là thứ không được lọt vào khóa luận.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from config import DATA_DIR, get_settings
from data.base import ObservationUnavailable, ProviderNotAvailable, get_observation_provider
from data.mock import MockObservationProvider
from evaluation import (
    EpisodeLabel,
    EpisodeLabelError,
    aggregate,
    confidence_breakdown,
    load_episodes,
    score_bundle,
)
from kb import get_knowledge_base
from narrator.narrator import DryRunNarrator
from pipeline import ExplanationPipeline, PipelineConfig
from rag.retrieve import build_retriever
from xai.base import AttributionNotAvailable, get_attribution_provider
from xai.mock import MockAttributionProvider


class SkipEpisode(Exception):
    """Episode không chạy được với cấu hình hiện tại — được liệt kê, không bị nuốt."""


def main() -> int:
    parser = argparse.ArgumentParser(description="Chấm điểm trên bộ episode gán nhãn")
    parser.add_argument("--episodes", default=None, help="Mặc định data/eval/episodes.yaml")
    parser.add_argument("--smoke", action="store_true", help="Dùng data/eval/smoke_episodes.yaml")
    parser.add_argument("--ablations", default="A,B,C,D,E")
    parser.add_argument("--rag", default="off", choices=["off", "qdrant", "memory"])
    parser.add_argument("--rag-min-score", type=float, default=None)
    parser.add_argument(
        "--xai-mode", default="aligned", choices=["aligned", "conflicting", "spurious"]
    )
    parser.add_argument("--out", default=None, help="Mặc định data/eval/results/<tên>.json")
    args = parser.parse_args()

    kb = get_knowledge_base()
    settings = get_settings()
    eval_dir = DATA_DIR / "eval"
    path = (
        Path(args.episodes)
        if args.episodes
        else eval_dir / ("smoke_episodes.yaml" if args.smoke else "episodes.yaml")
    )

    try:
        episodes = load_episodes(path, kb)
    except (FileNotFoundError, EpisodeLabelError) as exc:
        print(f"✗ {exc}")
        return 2

    print(f"Nhãn: {path}")
    print(
        f"  labeled {len(episodes.labeled)}  |  todo {episodes.todo}  |  rejected {episodes.rejected}"
    )
    if not episodes.labeled:
        print("\nChưa có episode nào ở status: labeled — chưa có gì để chấm.")
        return 1

    synthetic = sum(1 for e in episodes.labeled if e.synthetic)
    supervisor = sum(1 for e in episodes.labeled if e.supervisor_only())
    if synthetic:
        print(
            f"  ⚠ {synthetic} episode TỔNG HỢP — số liệu chỉ kiểm thử bộ chấm, KHÔNG dùng cho khóa luận"
        )
    if supervisor:
        print(f"  ⚠ {supervisor} episode chỉ dựa vào ý kiến GVHD — báo cáo tỉ lệ này")

    ablations = [a.strip().upper() for a in args.ablations.split(",") if a.strip()]
    retriever = build_retriever(args.rag, args.rag_min_score)

    scores = {level: [] for level in ablations}
    skipped: list[dict[str, str]] = []

    for label in episodes.labeled:
        try:
            observations, attributions = _providers_for(label, args.xai_mode, settings)
            for level in ablations:
                pipeline = ExplanationPipeline(
                    observation_provider=observations,
                    attribution_provider=attributions,
                    retriever=retriever,
                    narrator=DryRunNarrator(),
                    kb=kb,
                    config=PipelineConfig.ablation(level),
                )
                bundle = pipeline.build_bundle(
                    label.questions[0],
                    lat=label.lat,
                    lon=label.lon,
                    date=label.date,
                    step=label.step,
                    place_label=label.place,
                )
                scores[level].append(score_bundle(label, bundle, level))
        except (
            SkipEpisode,
            ObservationUnavailable,
            NotImplementedError,
            ProviderNotAvailable,
            AttributionNotAvailable,
        ) as exc:
            skipped.append({"episode_id": label.episode_id, "reason": str(exc).splitlines()[0]})
            for level in ablations:
                scores[level] = [s for s in scores[level] if s.episode_id != label.episode_id]

    _print_table(scores)
    if skipped:
        print(f"\n⚠ Bỏ qua {len(skipped)} episode:")
        for item in skipped:
            print(f"    {item['episode_id']}: {item['reason']}")

    out = (
        Path(args.out)
        if args.out
        else eval_dir
        / "results"
        / (f"{'smoke' if args.smoke else 'episodes'}_{datetime.now():%Y%m%d_%H%M%S}.json")
    )
    _write_results(out, args, episodes, retriever, scores, skipped, settings)
    print(f"\nChi tiết từng episode: {out}")
    return 0


_REAL_PROVIDERS: dict[str, tuple] = {}


def _providers_for(label: EpisodeLabel, xai_mode: str, settings):
    if label.synthetic:
        return MockObservationProvider(label.mock_episode), MockAttributionProvider(mode=xai_mode)

    if settings.data_backend == "mock":
        raise SkipEpisode(
            "episode thật nhưng EGXAQ_DATA_BACKEND=mock — cần dữ liệu thật (era5 hoặc lab)"
        )
    if "real" not in _REAL_PROVIDERS:
        observations = get_observation_provider()
        # SHAP giả dựng trên dữ liệu thật vẫn là SHAP giả → không đưa vào; verdict là NO_SHAP.
        attributions = get_attribution_provider() if settings.xai_backend != "mock" else None
        print(f"  Nguồn dữ liệu: {observations.name}")
        for note in getattr(observations, "missing_sources", []):
            print(f"  ⚠ {note}")
        _REAL_PROVIDERS["real"] = (observations, attributions)
    return _REAL_PROVIDERS["real"]


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _print_table(scores) -> None:
    print("\n" + "=" * 86)
    print(
        f"  {'cấu hình':<9}{'n':>4}  {'F1 chặt':>8}{'F1 nới':>8}{'macro F1':>9}"
        f"{'top-1':>7}{'khẳng định sai':>16}{'đúng loại':>11}"
    )
    print("  " + "-" * 82)
    for level, items in scores.items():
        s = aggregate(items)
        if not s.get("n"):
            print(f"  {level:<9}{0:>4}")
            continue
        print(
            f"  {level:<9}{s['n']:>4}  {_fmt(s['strict_f1']):>8}{_fmt(s['lenient_f1']):>8}"
            f"{_fmt(s['macro_strict_f1']):>9}{_fmt(s['top1_accuracy']):>7}"
            f"{_fmt(s['excluded_hit_rate']):>16}{_fmt(s['outcome_accuracy']):>11}"
        )
    print("=" * 86)
    print(
        "  khẳng định sai = tỉ lệ episode mà hệ thống khẳng định một cơ chế nguồn đã LOẠI TRỪ\n"
        "  đúng loại      = kể đúng câu chuyện: vì sao bẩn / vì sao sạch / không đủ căn cứ"
    )


def _write_results(out, args, episodes, retriever, scores, skipped, settings) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "episodes_path": str(episodes.path),
            "labeled": len(episodes.labeled),
            "todo": episodes.todo,
            "rejected": episodes.rejected,
            "synthetic": sum(1 for e in episodes.labeled if e.synthetic),
            "supervisor_only": sum(1 for e in episodes.labeled if e.supervisor_only()),
            "data_backend": settings.data_backend,
            "xai_backend": settings.xai_backend,
            "xai_mode": args.xai_mode,
            "rag": args.rag,
            "embedder": retriever.embedder.name if retriever else None,
            "min_score": retriever.config.min_score if retriever else None,
        },
        "summary": {level: aggregate(items) for level, items in scores.items()},
        "confidence": {level: confidence_breakdown(items) for level, items in scores.items()},
        "episodes": [s.to_dict() for items in scores.values() for s in items],
        "skipped": skipped,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
