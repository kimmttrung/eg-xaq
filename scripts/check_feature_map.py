"""Đối chiếu `knowledge/feature_map.yaml` với đặc trưng THẬT của mô hình lab.

    # kiểm tra nhanh bằng tên đặc trưng của mock (chạy được ngay, không cần lab)
    python scripts/check_feature_map.py --mock --allow-blind

    # ^ mock cần --allow-blind vì MECH_SOURCE_SECTOR_TRANSPORT chỉ kỳ vọng biến
    #   `wind_dir` mà mock không sinh đặc trưng hướng gió nào. Đó là hạn chế của
    #   mock, không phải của feature map. Với model lab thì BỎ cờ này đi.

    # khi đã có danh sách đặc trưng từ mentor (M4 trong data contract)
    python scripts/check_feature_map.py --names data/lab/feature_names.json
    python scripts/check_feature_map.py --names data/lab/features.txt

    # khi đã có checkpoint thật
    python scripts/check_feature_map.py --checkpoint-dir data/lab/models
    python scripts/check_feature_map.py --checkpoint-dir data/lab/models --step 3

CHẠY SCRIPT NÀY TRƯỚC KHI CHẠY PIPELINE TRÊN DATA LAB.

Lý do: nếu feature map chưa cập nhật, pipeline vẫn chạy trơn tru và vẫn cho ra
câu trả lời trông hợp lý — chỉ có điều `shap_agreement` bằng 0 ở mọi cơ chế và
tầng kiểm chứng chéo đã tắt lặng lẽ. Không có exception nào để mà nhìn thấy.
Script này biến lỗi im lặng đó thành một lỗi ồn ào.

Exit code 1 khi ánh xạ không đủ phủ → cắm được vào CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from kb import get_knowledge_base
from xai.coverage import (
    FeatureMapCoverageError,
    analyze_coverage,
    assert_mapping_healthy,
    format_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra feature_map.yaml có phủ hết đặc trưng của mô hình không"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mock", action="store_true", help="Dùng tên đặc trưng của mock provider")
    source.add_argument(
        "--names", help="File .json (list hoặc {step: list}) hoặc .txt mỗi dòng một tên"
    )
    source.add_argument("--checkpoint-dir", help="Thư mục chứa checkpoint XGBoost của lab")
    parser.add_argument("--step", type=int, default=None, help="Chỉ kiểm tra một bước dự báo")
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.90,
        help="Tỉ lệ đặc trưng nhận ra tối thiểu (mặc định 0.90)",
    )
    parser.add_argument(
        "--allow-blind",
        action="store_true",
        help="Không fail khi có cơ chế mù hoàn toàn (KHÔNG khuyến nghị)",
    )
    args = parser.parse_args()

    kb = get_knowledge_base()

    try:
        per_step = _collect_feature_names(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"✗ {exc}")
        return 2

    if not per_step:
        print("✗ Không đọc được tên đặc trưng nào.")
        return 2

    print(f"feature_map.yaml  version={kb.feature_map.version}  source={kb.feature_map.source}")
    if kb.feature_map.source == "mock" and not args.mock:
        print(
            '⚠ feature_map.yaml vẫn đang ghi source: "mock". '
            'Sau khi cập nhật theo lab nhớ đổi thành "lab".'
        )
    print()

    failed = False
    for step, names in sorted(per_step.items()):
        label = "tất cả" if step is None else f"t+{step}"
        print("=" * 72)
        print(f"Bước dự báo: {label}   ({len(names)} đặc trưng)")
        print("=" * 72)

        report = analyze_coverage(kb.feature_map, names)
        print(format_report(kb, report))
        print()

        try:
            assert_mapping_healthy(
                kb,
                names,
                min_recognized_ratio=args.min_ratio,
                allow_fully_blind=args.allow_blind,
            )
            print(f"✓ {label}: feature map đủ phủ.")
        except FeatureMapCoverageError as exc:
            failed = True
            print(f"✗ {label}: {exc}")
        print()

    if failed:
        print(
            "KẾT LUẬN: consistency check sẽ KHÔNG hoạt động đúng với ánh xạ hiện tại.\n"
            "Cập nhật knowledge/feature_map.yaml rồi chạy lại script này."
        )
        return 1

    print("KẾT LUẬN: ánh xạ đủ phủ, consistency check có căn cứ để chạy.")
    return 0


# =============================================================================
# Nguồn tên đặc trưng
# =============================================================================


def _collect_feature_names(args) -> dict[int | None, list[str]]:
    """Trả `{step: [tên đặc trưng]}`. Khóa `None` nghĩa là không phân theo step."""
    if args.mock:
        return {None: _mock_feature_names()}
    if args.names:
        return _names_from_file(Path(args.names), args.step)
    return _names_from_checkpoints(Path(args.checkpoint_dir), args.step)


def _mock_feature_names() -> list[str]:
    """Toàn bộ tên đặc trưng mà MockAttributionProvider CÓ THỂ sinh ra.

    Phải gộp qua mọi kịch bản, không lấy một lần dự báo. Attribution của một lần
    dự báo chỉ chứa đặc trưng có đóng góp khác 0 — ngày không có điểm cháy thì
    `fire_count_upwind` vắng mặt, và nếu lấy đúng ngày đó đi đánh giá feature map
    thì sẽ kết luận nhầm là biến `fire` không được phủ.

    Với model thật, danh sách tương ứng là `booster.feature_names` — luôn đầy đủ,
    không phụ thuộc lần dự báo nào.
    """
    from data.mock import EPISODES, HANOI_LAT, HANOI_LON, MockObservationProvider
    from xai.mock import MockAttributionProvider

    names: set[str] = set()
    for episode, spec in EPISODES.items():
        obs = MockObservationProvider(episode).get(HANOI_LAT, HANOI_LON, spec["date"], 0)
        attribution = MockAttributionProvider("aligned").get(
            HANOI_LAT, HANOI_LON, obs.date, 0, observation=obs
        )
        names |= {c.feature for c in attribution.contributions}
    return sorted(names)


def _names_from_file(path: Path, step: int | None) -> dict[int | None, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {None: [str(x) for x in data]}
        if isinstance(data, dict):
            per_step = {int(k): [str(x) for x in v] for k, v in data.items()}
            return {step: per_step[step]} if step is not None else per_step
        raise ValueError(f"{path}: JSON phải là list tên hoặc dict {{step: list}}")

    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return {None: [n for n in names if n and not n.startswith("#")]}


def _names_from_checkpoints(model_dir: Path, step: int | None) -> dict[int | None, list[str]]:
    """Đọc `feature_names` ra khỏi checkpoint XGBoost.

    Đây chính là thao tác mà `LabXGBAttributionProvider` sẽ phải làm lúc chạy thật
    (CLAUDE.md §8 bẫy #4): tên đặc trưng LUÔN đọc từ checkpoint, không bao giờ
    hardcode — vì XGBoost nhận ndarray mà không kiểm tra tên cột, nên sai thứ tự
    vẫn chạy và vẫn ra số đẹp.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục checkpoint: {model_dir}")

    patterns = ("*.json", "*.ubj", "*.pkl", "*.joblib", "*.model")
    files = sorted({f for pattern in patterns for f in model_dir.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"{model_dir} không có checkpoint nào (tìm: {', '.join(patterns)})")

    per_step: dict[int | None, list[str]] = {}
    for path in files:
        detected = _step_from_filename(path)
        if step is not None and detected != step:
            continue
        names = _feature_names_from_checkpoint(path)
        if names:
            per_step[detected] = names
        else:
            print(f"⚠ {path.name}: không đọc được feature_names — bỏ qua.")
    return per_step


def _step_from_filename(path: Path) -> int | None:
    """Suy bước dự báo từ tên file, ví dụ `model_step3.json` → 3.

    Quy ước đặt tên của lab là M1 trong data contract — chưa chốt. Nếu suy sai,
    script vẫn chạy và chỉ gộp chung, không ảnh hưởng kết luận về độ phủ.
    """
    import re

    match = re.search(r"(?:step|t\+?|h)(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _feature_names_from_checkpoint(path: Path) -> list[str] | None:
    """Lấy danh sách đặc trưng từ một file checkpoint, thử vài dạng đóng gói."""
    if path.suffix.lower() in {".json", ".ubj", ".model"}:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "Đọc checkpoint cần xgboost. Cài: pip install -r requirements-lab.txt"
            ) from exc
        booster = xgb.Booster()
        booster.load_model(str(path))
        return list(booster.feature_names or [])

    import pickle

    with path.open("rb") as fh:
        obj = pickle.load(fh)

    # sklearn API
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]

    # XGBModel → Booster
    get_booster = getattr(obj, "get_booster", None)
    if callable(get_booster):
        return list(get_booster().feature_names or [])

    # Booster trần
    names = getattr(obj, "feature_names", None)
    if names:
        return [str(n) for n in names]

    # dict {"model": ..., "features": [...]}
    if isinstance(obj, dict):
        for key in ("feature_names", "features", "columns"):
            if key in obj:
                return [str(n) for n in obj[key]]

    return None


if __name__ == "__main__":
    raise SystemExit(main())
