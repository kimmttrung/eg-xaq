"""Kiểm tra độ phủ của `knowledge/feature_map.yaml` so với đặc trưng thật của lab.

VÌ SAO MODULE NÀY TỒN TẠI
=========================

Consistency check (đóng góp nghiên cứu số 2) chỉ chạy được khi tên đặc trưng của
mô hình lab ánh xạ được về biến chuẩn hóa. Nếu `feature_map.yaml` chưa cập nhật
theo danh sách đặc trưng thật, `FeatureMap.resolve()` trả `None` cho mọi thứ và:

    shap_agreement  = 0.0 cho MỌI cơ chế
    verdict         = PARTIAL cho MỌI cơ chế đang kích hoạt
    CONFLICT        = không bao giờ phát hiện được
    MODEL_ONLY      = không bao giờ phát hiện được

Điều nguy hiểm là hệ thống KHÔNG ném lỗi. Nó vẫn chấm điểm, vẫn xếp hạng, vẫn
sinh ra câu trả lời trông rất tự tin — chỉ có điều tầng kiểm chứng chéo đã tắt
lặng lẽ. Đây là kiểu hỏng không giống hỏng, nên phải có hàng rào phát hiện chủ
động thay vì chờ nhìn ra qua kết quả.

Module cố ý KHÔNG import xgboost/shap: nó chỉ nhận vào một danh sách TÊN đặc
trưng. Nhờ vậy test chạy offline được, và script kiểm tra checkpoint thật chỉ
cần lo phần đọc tên ra khỏi model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kb import FeatureMap, KnowledgeBase


class FeatureMapCoverageError(RuntimeError):
    """Feature map không phủ đủ đặc trưng của mô hình → consistency check vô hiệu."""


@dataclass(frozen=True)
class CoverageReport:
    """Kết quả đối chiếu danh sách đặc trưng với feature map.

    Ba nhóm, phân biệt rạch ròi:

    - `resolved`        — ánh xạ được về biến chuẩn hóa. Đây là phần nuôi sống
                          consistency check.
    - `non_mechanistic` — nhận ra được và CỐ Ý đánh dấu là không mang cơ chế vật lý
                          (toạ độ, mã thời gian, biến tĩnh theo pixel). Không phải
                          lỗi: chúng được đếm vào `non_mechanistic_share`.
    - `unmapped`        — KHÔNG nhận ra. Đây mới là thứ đáng báo động: mỗi tên ở
                          đây là một phần attribution bị consistency check bỏ qua
                          hoàn toàn, không ai biết.
    """

    resolved: dict[str, str]
    non_mechanistic: tuple[str, ...]
    unmapped: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.resolved) + len(self.non_mechanistic) + len(self.unmapped)

    @property
    def recognized_ratio(self) -> float:
        """Tỉ lệ đặc trưng NHẬN RA được (dù là cơ chế hay phi cơ chế)."""
        return (len(self.resolved) + len(self.non_mechanistic)) / self.total if self.total else 0.0

    @property
    def mechanistic_ratio(self) -> float:
        """Tỉ lệ đặc trưng ánh xạ được về biến cơ chế — phần thực sự nuôi consistency."""
        return len(self.resolved) / self.total if self.total else 0.0

    @property
    def covered_variables(self) -> set[str]:
        """Các biến chuẩn hóa có ít nhất một đặc trưng trỏ tới."""
        return set(self.resolved.values())


def analyze_coverage(feature_map: FeatureMap, feature_names: Iterable[str]) -> CoverageReport:
    """Phân loại từng tên đặc trưng theo feature map."""
    resolved: dict[str, str] = {}
    non_mechanistic: list[str] = []
    unmapped: list[str] = []

    for name in feature_names:
        canonical = feature_map.resolve(name)
        if canonical is not None:
            resolved[name] = canonical
        elif feature_map.is_non_mechanistic(name):
            non_mechanistic.append(name)
        else:
            unmapped.append(name)

    return CoverageReport(
        resolved=resolved,
        non_mechanistic=tuple(non_mechanistic),
        unmapped=tuple(unmapped),
    )


def mechanism_blind_spots(kb: KnowledgeBase, report: CoverageReport) -> dict[str, list[str]]:
    """Cơ chế nào có biến kỳ vọng mà KHÔNG đặc trưng nào trỏ tới.

    Đây là chỉ số sắc hơn tỉ lệ phủ thô. Ví dụ: nếu không đặc trưng nào ánh xạ về
    `blh`, thì mọi cơ chế dựa vào PBLH sẽ vĩnh viễn có `shap_agreement = 0` —
    ngay cả khi 90% đặc trưng khác ánh xạ tốt và tỉ lệ phủ trông vẫn đẹp.

    Trả về `{mechanism_id: [biến bị mù]}`, chỉ gồm cơ chế có ít nhất một biến mù.
    """
    covered = report.covered_variables
    blind: dict[str, list[str]] = {}
    for mech in kb.mechanisms.values():
        missing = sorted(var for var in mech.variables if var not in covered)
        if missing:
            blind[mech.id] = missing
    return blind


def fully_blind_mechanisms(kb: KnowledgeBase, report: CoverageReport) -> list[str]:
    """Cơ chế mà TOÀN BỘ biến kỳ vọng đều không được phủ.

    Những cơ chế này không bao giờ đạt verdict CONFIRMED hay CONFLICT được nữa —
    chúng bị khóa cứng ở PARTIAL. Nếu danh sách này không rỗng thì đóng góp nghiên
    cứu số 2 đã mất hiệu lực trên đúng những cơ chế đó.
    """
    blind = mechanism_blind_spots(kb, report)
    return sorted(
        mech_id
        for mech_id, missing in blind.items()
        if kb.mechanisms[mech_id].variables
        and len(missing) == len(kb.mechanisms[mech_id].variables)
    )


def assert_mapping_healthy(
    kb: KnowledgeBase,
    feature_names: Iterable[str],
    min_recognized_ratio: float = 0.90,
    allow_fully_blind: bool = False,
) -> CoverageReport:
    """Ném lỗi nếu feature map không đủ phủ để consistency check có ý nghĩa.

    Dùng trong test (khóa hành vi hiện tại) và trong script kiểm tra checkpoint
    thật (chặn trước khi chạy cả pipeline trên một ánh xạ hỏng).

    `min_recognized_ratio` đặt ở 0.90 chứ không phải 1.0 vì mô hình lab có thể có
    vài đặc trưng phụ mà ta chưa quyết định xếp vào đâu. Nhưng `fully_blind` thì
    mặc định KHÔNG được phép: một cơ chế mù hoàn toàn là hỏng có thật, không phải
    sai số chấp nhận được.
    """
    report = analyze_coverage(kb.feature_map, feature_names)

    problems: list[str] = []
    if report.recognized_ratio < min_recognized_ratio:
        problems.append(
            f"Chỉ nhận ra {report.recognized_ratio:.0%} đặc trưng "
            f"(cần ≥ {min_recognized_ratio:.0%}). Chưa ánh xạ: "
            f"{', '.join(report.unmapped[:12])}" + (" …" if len(report.unmapped) > 12 else "")
        )

    if not allow_fully_blind:
        blind = fully_blind_mechanisms(kb, report)
        if blind:
            problems.append(
                "Cơ chế bị mù hoàn toàn (không biến nào được phủ → verdict khóa cứng ở "
                f"PARTIAL, không bao giờ phát hiện được CONFLICT): {', '.join(blind)}"
            )

    if problems:
        raise FeatureMapCoverageError(
            "knowledge/feature_map.yaml không phủ đủ đặc trưng của mô hình:\n  - "
            + "\n  - ".join(problems)
            + "\n\nSửa feature_map.yaml rồi chạy lại. "
            "Xem docs/02-data-contract.md §7.3 và scripts/check_feature_map.py."
        )

    return report


def format_report(kb: KnowledgeBase, report: CoverageReport) -> str:
    """Bản in cho người đọc — dùng trong scripts/check_feature_map.py."""
    lines: list[str] = []
    lines.append(f"Tổng số đặc trưng          : {report.total}")
    lines.append(
        f"  ánh xạ về biến cơ chế    : {len(report.resolved)} ({report.mechanistic_ratio:.0%})"
    )
    lines.append(f"  phi cơ chế (cố ý)        : {len(report.non_mechanistic)}")
    lines.append(f"  CHƯA ÁNH XẠ              : {len(report.unmapped)}")
    lines.append("")

    if report.resolved:
        lines.append("Ánh xạ được:")
        by_variable: dict[str, list[str]] = {}
        for feature, canonical in sorted(report.resolved.items()):
            by_variable.setdefault(canonical, []).append(feature)
        for canonical in sorted(by_variable):
            lines.append(f"  {canonical:<14} ← {', '.join(by_variable[canonical])}")
        lines.append("")

    if report.non_mechanistic:
        lines.append(f"Phi cơ chế: {', '.join(sorted(report.non_mechanistic))}")
        lines.append("")

    if report.unmapped:
        lines.append("⚠ CHƯA ÁNH XẠ — mỗi tên dưới đây là một phần attribution bị bỏ qua:")
        for name in sorted(report.unmapped):
            lines.append(f"  {name}")
        lines.append("")

    blind = mechanism_blind_spots(kb, report)
    if blind:
        lines.append("⚠ Cơ chế có biến không được phủ:")
        for mech_id in sorted(blind):
            total_vars = len(kb.mechanisms[mech_id].variables)
            missing = blind[mech_id]
            tag = "  ← MÙ HOÀN TOÀN" if len(missing) == total_vars else ""
            lines.append(f"  {mech_id:<32} thiếu: {', '.join(missing)}{tag}")
    else:
        lines.append("✓ Mọi biến mà cơ chế kỳ vọng đều có đặc trưng trỏ tới.")

    return "\n".join(lines)
