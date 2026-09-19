"""Bộ đánh giá — nhãn nguyên nhân cho episode và cách chấm điểm.

Thiết kế: docs/06-evaluation.md §2–§3. Dữ liệu: data/eval/episodes.yaml.
Hướng dẫn gán nhãn: data/eval/LABELING.md.

NGUYÊN TẮC ĐỘC LẬP NHÃN
=======================
Nhãn nguyên nhân PHẢI đến từ nguồn độc lập với hệ thống: báo cáo CEM, bài báo phân
tích đúng đợt đó, bản tin có dẫn chuyên gia. Không được suy nhãn từ chính rule của
hệ thống, hay từ số khí tượng mà rule đọc vào. Làm vậy thì Cause F1 chỉ đo hệ thống
có khớp với chính nó không — con số cao đến mấy cũng vô nghĩa.

Vì cùng lý do đó, NGÀY được chọn bằng PM2.5 quan trắc (`select_candidates`), không
chọn theo điều kiện khí tượng: chọn theo PBLH thấp rồi kiểm tra xem hệ thống có ra
"PBLH thấp" không là vòng lặp.

Module này không import provider hay RAG — nó chỉ nhận `EvidenceBundle` đã dựng sẵn.
Việc chạy pipeline nằm ở scripts/score_episodes.py.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kb import KnowledgeBase
from narrator.narrator import story_of
from schemas import EvidenceBundle

SourceType = Literal[
    "cem_report",  # báo cáo CEM / cơ quan quan trắc — ưu tiên cao nhất
    "peer_reviewed",  # bài báo phân tích ĐÚNG đợt đó
    "forecast_bulletin",  # bản tin dự báo chất lượng không khí
    "news_expert",  # báo chí có dẫn nguồn chuyên gia
    "supervisor",  # ý kiến GVHD — phương án cuối, được thống kê riêng
    "synthetic_design",  # chỉ dành cho episode tổng hợp dùng kiểm thử
]
Outcome = Literal["cause", "clean", "insufficient"]
Group = Literal["polluted_winter", "polluted_transition", "clean", "other"]

WINTER_MONTHS = frozenset({11, 12, 1, 2})
TRANSITION_MONTHS = frozenset({3, 4, 9, 10})
DRY_SEASON_MONTHS = frozenset({11, 12, 1, 2, 3, 4})


class EpisodeLabelError(ValueError):
    """File nhãn không hợp lệ. Liệt kê MỌI lỗi một lần, không dừng ở lỗi đầu tiên."""


# =============================================================================
# Schema nhãn
# =============================================================================


class LabelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SourceType
    citation: str = Field(min_length=3, description="Tên báo cáo / bài báo / bản tin")
    locator: str | None = Field(None, description="URL, DOI, hoặc trang/mục cụ thể")


class EpisodeLabel(BaseModel):
    """Một ngày đã gán nhãn nguyên nhân.

    Ba danh sách cơ chế có nghĩa khác nhau, đừng gộp:

    - `primary_mechanisms`      — nguồn nói RÕ đây là nguyên nhân chính. Hệ thống
                                  phải tìm ra; bỏ sót là lỗi.
    - `contributing_mechanisms` — nguồn nhắc tới như yếu tố góp phần. Hệ thống nêu
                                  ra không bị phạt ở chế độ nới, cũng không được thưởng.
    - `excluded_mechanisms`     — nguồn LOẠI TRỪ rõ ràng (ví dụ "không phải do đốt
                                  rơm rạ"). Hệ thống khẳng định cơ chế này là lỗi nặng
                                  nhất, được đếm riêng.

    Cơ chế không nằm trong danh sách nào = nguồn không nói tới.
    """

    model_config = ConfigDict(extra="forbid")  # gõ sai tên trường → báo lỗi ngay

    episode_id: str = Field(min_length=1)
    status: Literal["labeled"] = "labeled"
    date: Date
    place: str = "Hà Nội"
    lat: float = 21.03
    lon: float = 105.85
    step: int = Field(0, ge=0)
    group: Group
    expected_outcome: Outcome
    questions: list[str] = Field(min_length=1)

    primary_mechanisms: list[str] = Field(default_factory=list)
    contributing_mechanisms: list[str] = Field(default_factory=list)
    excluded_mechanisms: list[str] = Field(default_factory=list)

    rationale: str = Field(min_length=10, description="Tóm tắt nguồn nói gì, bằng lời của bạn")
    sources: list[LabelSource] = Field(min_length=1)
    annotator: str = Field(min_length=1)
    label_confidence: Literal["high", "medium", "low"]

    pm25_obs_ugm3: float | None = None
    synthetic: bool = False
    mock_episode: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> EpisodeLabel:
        problems: list[str] = []

        lists = {
            "primary": set(self.primary_mechanisms),
            "contributing": set(self.contributing_mechanisms),
            "excluded": set(self.excluded_mechanisms),
        }
        names = list(lists)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                both = lists[first] & lists[second]
                if both:
                    problems.append(f"{first} và {second} trùng nhau: {sorted(both)}")

        if self.expected_outcome in ("cause", "clean") and not self.primary_mechanisms:
            problems.append("expected_outcome là cause/clean thì phải có primary_mechanisms")
        if self.expected_outcome == "insufficient" and self.primary_mechanisms:
            problems.append("expected_outcome là insufficient thì không được có primary_mechanisms")

        has_synthetic_source = any(s.type == "synthetic_design" for s in self.sources)
        if self.synthetic:
            if not self.mock_episode:
                problems.append("episode tổng hợp phải khai mock_episode")
        else:
            if self.mock_episode:
                problems.append(
                    "episode thật không được dùng mock_episode — dữ liệu giả cho ra số vô nghĩa"
                )
            if has_synthetic_source:
                problems.append("nguồn synthetic_design chỉ dành cho episode tổng hợp")

        if problems:
            raise ValueError("; ".join(problems))
        return self

    def supervisor_only(self) -> bool:
        """Nhãn chỉ dựa vào ý kiến GVHD — thống kê riêng, hội đồng sẽ hỏi tỉ lệ này."""
        return all(s.type == "supervisor" for s in self.sources)


def validate_against_kb(label: EpisodeLabel, kb: KnowledgeBase) -> list[str]:
    """Kiểm tra nhãn có khớp tri thức hiện có không."""
    problems: list[str] = []
    for field in ("primary_mechanisms", "contributing_mechanisms", "excluded_mechanisms"):
        for mech_id in getattr(label, field):
            if mech_id not in kb.mechanisms:
                problems.append(f"{label.episode_id}: {field} có cơ chế không tồn tại '{mech_id}'")

    primary = [kb.mechanisms[m] for m in label.primary_mechanisms if m in kb.mechanisms]
    if label.expected_outcome == "clean" and any(m.effect != "decrease" for m in primary):
        problems.append(
            f"{label.episode_id}: ngày sạch thì nguyên nhân chính phải là cơ chế LÀM GIẢM "
            "PM2.5 (effect: decrease)"
        )
    if (
        label.expected_outcome == "cause"
        and primary
        and all(m.effect == "decrease" for m in primary)
    ):
        problems.append(
            f"{label.episode_id}: episode ô nhiễm cần ít nhất một cơ chế LÀM TĂNG PM2.5"
        )
    return problems


@dataclass(frozen=True)
class EpisodeSet:
    labeled: list[EpisodeLabel]
    todo: int
    rejected: int
    path: Path


def load_episodes(path: Path, kb: KnowledgeBase) -> EpisodeSet:
    """Nạp file nhãn. Mục `status: todo` / `rejected` được đếm rồi bỏ qua.

    Fail fast như kb.py: gom mọi lỗi rồi ném một lần. Một nhãn sai tên cơ chế mà bị
    bỏ qua lặng lẽ sẽ làm lệch Cause F1 mà không ai biết.
    """
    if not path.exists():
        raise FileNotFoundError(f"Chưa có file nhãn: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("episodes") if isinstance(raw, dict) else raw
    if items is None:
        items = []
    if not isinstance(items, list):
        raise EpisodeLabelError(f"{path}: cần khóa `episodes:` chứa một danh sách")

    problems: list[str] = []
    labeled: list[EpisodeLabel] = []
    seen: set[str] = set()
    todo = rejected = 0

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            problems.append(f"mục #{index}: không phải một mapping")
            continue
        ident = str(item.get("episode_id") or f"#{index}")
        status = item.get("status", "labeled")

        if status == "todo":
            todo += 1
            continue
        if status == "rejected":
            if not item.get("reject_reason"):
                problems.append(f"{ident}: status rejected thì phải ghi reject_reason")
            rejected += 1
            continue
        if status != "labeled":
            problems.append(f"{ident}: status không hợp lệ '{status}' (todo | labeled | rejected)")
            continue

        try:
            label = EpisodeLabel.model_validate(item)
        except ValidationError as exc:
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"]) or "nhãn"
                problems.append(f"{ident}: {where} — {error['msg']}")
            continue

        if label.episode_id in seen:
            problems.append(f"{ident}: episode_id bị trùng")
            continue
        seen.add(label.episode_id)
        problems.extend(validate_against_kb(label, kb))
        labeled.append(label)

    if problems:
        raise EpisodeLabelError(f"{path} có {len(problems)} lỗi:\n  - " + "\n  - ".join(problems))
    return EpisodeSet(labeled=labeled, todo=todo, rejected=rejected, path=path)


# =============================================================================
# Chấm điểm
# =============================================================================


def predicted_mechanisms(bundle: EvidenceBundle) -> list[str]:
    """Mọi cơ chế hệ thống KHẲNG ĐỊNH, xếp theo điểm giảm dần (cả tăng lẫn giảm PM2.5)."""
    ranked = sorted([*bundle.hypotheses, *bundle.suppressors], key=lambda h: h.score, reverse=True)
    return [h.mechanism_id for h in ranked]


def predicted_outcome(bundle: EvidenceBundle) -> Outcome:
    """Câu chuyện hệ thống KỂ cho người dùng: vì sao bẩn, vì sao sạch, hay không đủ căn cứ.

    Dùng đúng hàm narrator dùng để chọn câu mở đầu (`narrator.story_of`), nên metric đo
    thứ người dùng thực sự đọc. Không định nghĩa lại ở đây: hai định nghĩa lệch nhau thì
    metric và câu trả lời sẽ nói hai chuyện khác nhau.
    """
    return story_of(bundle)


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    config: str
    predicted: tuple[str, ...]
    predicted_outcome: str
    expected_outcome: str
    primary: frozenset[str]
    contributing: frozenset[str]
    excluded: frozenset[str]
    top_confidence: str | None
    overall_confidence: str

    def counts(self, lenient: bool) -> tuple[int, int, int]:
        """(tp, fp, fn) trên cơ chế.

        Chặt : chỉ `primary` là đúng; nêu cơ chế góp phần cũng bị tính là dương tính giả.
        Nới  : cơ chế góp phần không bị phạt, cũng không được thưởng.
        Recall ở cả hai chế độ chỉ tính trên `primary` — hệ thống phải tìm ra nguyên
        nhân chính, không bắt buộc liệt kê hết yếu tố góp phần.
        """
        predicted = set(self.predicted)
        allowed = self.primary | self.contributing if lenient else self.primary
        return (
            len(predicted & self.primary),
            len(predicted - allowed),
            len(self.primary - predicted),
        )

    @property
    def excluded_hits(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.predicted) & self.excluded))

    @property
    def top1_hit(self) -> bool | None:
        """Cơ chế xếp đầu có phải nguyên nhân chính không. None nếu nhãn không có primary."""
        if not self.primary:
            return None
        return bool(self.predicted) and self.predicted[0] in self.primary

    @property
    def outcome_correct(self) -> bool:
        return self.predicted_outcome == self.expected_outcome

    def to_dict(self) -> dict[str, Any]:
        strict = self.counts(lenient=False)
        lenient = self.counts(lenient=True)
        return {
            "episode_id": self.episode_id,
            "config": self.config,
            "predicted": list(self.predicted),
            "predicted_outcome": self.predicted_outcome,
            "expected_outcome": self.expected_outcome,
            "primary": sorted(self.primary),
            "contributing": sorted(self.contributing),
            "excluded": sorted(self.excluded),
            "strict_tp_fp_fn": list(strict),
            "lenient_tp_fp_fn": list(lenient),
            "excluded_hits": list(self.excluded_hits),
            "top1_hit": self.top1_hit,
            "top_confidence": self.top_confidence,
            "overall_confidence": self.overall_confidence,
        }


def score_bundle(label: EpisodeLabel, bundle: EvidenceBundle, config: str) -> EpisodeScore:
    ranked = sorted([*bundle.hypotheses, *bundle.suppressors], key=lambda h: h.score, reverse=True)
    return EpisodeScore(
        episode_id=label.episode_id,
        config=config,
        predicted=tuple(h.mechanism_id for h in ranked),
        predicted_outcome=predicted_outcome(bundle),
        expected_outcome=label.expected_outcome,
        primary=frozenset(label.primary_mechanisms),
        contributing=frozenset(label.contributing_mechanisms),
        excluded=frozenset(label.excluded_mechanisms),
        top_confidence=ranked[0].confidence.value if ranked else None,
        overall_confidence=bundle.overall_confidence.value,
    )


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate(scores: Iterable[EpisodeScore]) -> dict[str, float | int | None]:
    """Gộp điểm một cấu hình.

    Dùng MICRO-average làm số chính: cộng tp/fp/fn qua mọi episode rồi mới tính. Với
    30–50 episode, macro-average dao động mạnh vì mỗi episode chỉ có 1–3 nguyên nhân.
    Macro vẫn được báo cáo kèm để thấy độ lệch giữa các episode.
    """
    scores = list(scores)
    if not scores:
        return {"n": 0}

    def summed(lenient: bool) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for score in scores:
            a, b, c = score.counts(lenient)
            tp, fp, fn = tp + a, fp + b, fn + c
        return tp, fp, fn

    sp, sr, sf = _prf(*summed(lenient=False))
    lp, lr, lf = _prf(*summed(lenient=True))

    with_primary = [s for s in scores if s.primary]
    macro = (
        sum(_prf(*s.counts(lenient=False))[2] for s in with_primary) / len(with_primary)
        if with_primary
        else None
    )
    top1 = sum(1 for s in with_primary if s.top1_hit) / len(with_primary) if with_primary else None

    return {
        "n": len(scores),
        "strict_precision": sp,
        "strict_recall": sr,
        "strict_f1": sf,
        "lenient_precision": lp,
        "lenient_recall": lr,
        "lenient_f1": lf,
        "macro_strict_f1": macro,
        "top1_accuracy": top1,
        "excluded_hit_rate": sum(1 for s in scores if s.excluded_hits) / len(scores),
        "outcome_accuracy": sum(1 for s in scores if s.outcome_correct) / len(scores),
    }


def confidence_breakdown(scores: Iterable[EpisodeScore]) -> dict[str, dict[str, float | int]]:
    """Top-1 đúng bao nhiêu phần, tách theo nhãn tin cậy hệ thống tự báo.

    Hệ thống báo CAO mà top-1 sai nhiều hơn khi báo THẤP → độ tin cậy không hiệu
    chỉnh, và đó là phát hiện phải báo cáo.
    """
    groups: dict[str, list[EpisodeScore]] = defaultdict(list)
    for score in scores:
        if score.primary:
            groups[score.top_confidence or "không khẳng định gì"].append(score)
    return {
        label: {"n": len(items), "top1_accuracy": sum(1 for s in items if s.top1_hit) / len(items)}
        for label, items in groups.items()
    }


# =============================================================================
# Chọn ngày ứng viên theo PM2.5 quan trắc
# =============================================================================


@dataclass(frozen=True)
class DailyRecord:
    date: Date
    pm25_ugm3: float


@dataclass(frozen=True)
class Candidate:
    date: Date
    group: Group
    pm25_ugm3: float
    reason: str


def select_candidates(
    records: Iterable[DailyRecord],
    per_group: int = 15,
    min_gap_days: int = 5,
) -> list[Candidate]:
    """Chọn ngày ứng viên CHỈ dựa vào PM2.5 quan trắc, phân tầng theo docs/06 §2.1.

    - polluted_winter     : PM2.5 cao nhất tháng 11–2
    - polluted_transition : PM2.5 cao nhất tháng 3–4, 9–10
    - clean               : PM2.5 thấp nhất, nửa mùa khô (ứng viên gió mùa ĐB) + nửa
                            mùa mưa (ứng viên rửa trôi)

    `min_gap_days` áp cho MỌI ngày đã chọn: một đợt ô nhiễm kéo dài 5 ngày chỉ được
    đại diện bởi một ngày. Không có ràng buộc này, top-15 mùa đông thường chỉ là 3–4
    đợt lặp lại, và bộ đánh giá đo đi đo lại cùng một tình huống.

    Hoàn toàn xác định: hòa điểm thì xếp theo ngày.
    """
    pool = list(records)
    chosen: list[Date] = []
    result: list[Candidate] = []

    def take(candidates: list[DailyRecord], n: int, group: Group, reason: str) -> None:
        taken = 0
        for record in candidates:
            if taken >= n:
                break
            if all(abs((record.date - other).days) >= min_gap_days for other in chosen):
                chosen.append(record.date)
                result.append(Candidate(record.date, group, record.pm25_ugm3, reason))
                taken += 1

    highest = sorted(pool, key=lambda r: (-r.pm25_ugm3, r.date))
    lowest = sorted(pool, key=lambda r: (r.pm25_ugm3, r.date))

    take(
        [r for r in highest if r.date.month in WINTER_MONTHS],
        per_group,
        "polluted_winter",
        "PM2.5 cao nhất tháng 11–2",
    )
    take(
        [r for r in highest if r.date.month in TRANSITION_MONTHS],
        per_group,
        "polluted_transition",
        "PM2.5 cao nhất tháng 3–4, 9–10",
    )
    dry = (per_group + 1) // 2
    take(
        [r for r in lowest if r.date.month in DRY_SEASON_MONTHS],
        dry,
        "clean",
        "PM2.5 thấp nhất mùa khô — ứng viên gió mùa Đông Bắc",
    )
    take(
        [r for r in lowest if r.date.month not in DRY_SEASON_MONTHS],
        per_group - dry,
        "clean",
        "PM2.5 thấp nhất mùa mưa — ứng viên rửa trôi",
    )
    return sorted(result, key=lambda c: c.date)


_DEFAULT_QUESTION = {
    "polluted_winter": "Vì sao hôm nay Hà Nội ô nhiễm nặng?",
    "polluted_transition": "Vì sao chất lượng không khí Hà Nội xấu đột ngột hôm nay?",
    "clean": "Vì sao hôm nay không khí Hà Nội trong lành?",
    "other": "Vì sao chất lượng không khí Hà Nội hôm nay như vậy?",
}


def render_worksheet(
    candidates: list[Candidate],
    kb: KnowledgeBase,
    data_note: str,
    second_annotator: bool = False,
) -> str:
    """Sinh file YAML để gán nhãn tay. Mọi mục bắt đầu ở `status: todo`.

    Cố ý KHÔNG in PBLH, gió, mưa của ngày đó: người gán nhãn nhìn thấy "PBLH 320 m"
    sẽ có xu hướng ghi luôn "lớp xáo trộn thấp" — tức là tái tạo lại rule, không phải
    đọc nguồn. Nhãn phải đến từ việc đọc báo cáo.
    """
    mech_lines = [
        f"#   {m.id:<30} {'↑' if m.effect == 'increase' else '↓'} {m.name}"
        for m in kb.mechanisms.values()
    ]
    header = [
        "# =============================================================================",
        "# Bộ episode đánh giá EG-XAQ — PHIẾU GÁN NHÃN",
        "# =============================================================================",
        "# Đọc data/eval/LABELING.md trước khi gán nhãn.",
        "#",
        f"# Ngày được chọn tự động theo PM2.5 quan trắc: {data_note}",
        "# Nhãn nguyên nhân PHẢI lấy từ nguồn độc lập (CEM, bài báo, bản tin có chuyên gia).",
        "# Không tìm được nguồn → đổi status thành rejected và ghi reject_reason.",
        "#",
        "# ⚠ Hỏi mentor (M10): các ngày này có nằm trong tập huấn luyện của mô hình lab không.",
        "#",
        "# Cơ chế hợp lệ (↑ làm tăng PM2.5, ↓ làm giảm):",
        *mech_lines,
        "",
        "episodes:",
    ]
    if second_annotator:
        header.insert(
            3,
            "# ⚠ PHIẾU NGƯỜI GÁN NHÃN THỨ HAI — gán ĐỘC LẬP, KHÔNG mở phiếu của người kia.",
        )
    body: list[str] = []
    for c in candidates:
        body += [
            f"  - episode_id: hn-{c.date.isoformat()}",
            "    status: todo              # todo | labeled | rejected",
            f"    date: {c.date.isoformat()}",
            f"    group: {c.group:<20}# chọn vì: {c.reason}",
            f"    pm25_obs_ugm3: {c.pm25_ugm3:g}",
            "    expected_outcome:         # cause | clean | insufficient",
            "    questions:",
            f"      - {_DEFAULT_QUESTION[c.group]}",
            "    primary_mechanisms: []",
            "    contributing_mechanisms: []",
            "    excluded_mechanisms: []",
            '    rationale: ""',
            "    sources: []               # - {type: cem_report, citation: ..., locator: ...}",
            '    annotator: ""',
            "    label_confidence:         # high | medium | low",
            "",
        ]
    return "\n".join(header + body)


def sample_for_second_annotator(
    candidates: list[Candidate], n: int = 15, seed: int = 42
) -> list[Candidate]:
    """Chọn tập con cho người gán nhãn thứ hai, rải đều qua các nhóm.

    Rải theo nhóm vì κ tính trên một nhóm duy nhất (ví dụ toàn ngày sạch) không nói lên
    độ khó thật của việc gán nhãn. Nhận `seed` để tái lập được.
    """
    rng = random.Random(seed)
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    for group in sorted(by_group):
        rng.shuffle(by_group[group])

    target = min(n, len(candidates))
    picked: list[Candidate] = []
    while len(picked) < target:
        for group in sorted(by_group):
            if by_group[group] and len(picked) < target:
                picked.append(by_group[group].pop())
    return sorted(picked, key=lambda c: c.date)


# =============================================================================
# Độ đồng thuận giữa hai người gán nhãn
# =============================================================================


def mechanism_role(label: EpisodeLabel, mechanism_id: str) -> str:
    if mechanism_id in label.primary_mechanisms:
        return "primary"
    if mechanism_id in label.contributing_mechanisms:
        return "contributing"
    if mechanism_id in label.excluded_mechanisms:
        return "excluded"
    return "none"


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """κ = (p_o − p_e) / (1 − p_e). None khi không xác định (p_e = 1, hoặc không có cặp)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    expected = sum(count_a[c] * count_b[c] for c in set(count_a) | set(count_b)) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class Agreement:
    common: int
    only_first: int
    only_second: int
    mechanism_kappa: float | None
    outcome_kappa: float | None
    disagreements: list[str]


def label_agreement(first: EpisodeSet, second: EpisodeSet, kb: KnowledgeBase) -> Agreement:
    """Cohen's κ giữa hai người gán nhãn, trên các episode cả hai đều đã `labeled`.

    κ cơ chế tính trên mọi cặp (episode × cơ chế), mỗi cặp một vai trò: primary /
    contributing / excluded / none. Phần lớn cặp là none/none nên tỉ lệ khớp thô luôn
    cao — đó là lý do báo cáo κ (đã trừ phần khớp do ngẫu nhiên) chứ không báo cáo %.
    """
    by_first = {e.episode_id: e for e in first.labeled}
    by_second = {e.episode_id: e for e in second.labeled}
    common = sorted(set(by_first) & set(by_second))

    mechanism_pairs: list[tuple[str, str]] = []
    outcome_pairs: list[tuple[str, str]] = []
    disagreements: list[str] = []
    for ident in common:
        a, b = by_first[ident], by_second[ident]
        outcome_pairs.append((a.expected_outcome, b.expected_outcome))
        if a.expected_outcome != b.expected_outcome:
            disagreements.append(
                f"{ident} expected_outcome: {a.expected_outcome} ≠ {b.expected_outcome}"
            )
        for mechanism_id in kb.mechanisms:
            role_a, role_b = mechanism_role(a, mechanism_id), mechanism_role(b, mechanism_id)
            mechanism_pairs.append((role_a, role_b))
            if role_a != role_b:
                disagreements.append(f"{ident} {mechanism_id}: {role_a} ≠ {role_b}")

    return Agreement(
        common=len(common),
        only_first=len(set(by_first) - set(by_second)),
        only_second=len(set(by_second) - set(by_first)),
        mechanism_kappa=cohens_kappa(mechanism_pairs),
        outcome_kappa=cohens_kappa(outcome_pairs),
        disagreements=disagreements,
    )
