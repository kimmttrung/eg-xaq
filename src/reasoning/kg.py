"""Knowledge Graph cơ chế khí quyển–ô nhiễm.

KG ở đây cố ý NHẸ: không ontology đồ sộ, không Neo4j, chỉ một `networkx.DiGraph`
dựng từ chính hai file YAML. Nó làm đúng ba việc:

1. **Truy vết (provenance)** — mỗi cạnh nhân quả gắn với rule_id và/hoặc citation,
   nên mọi tuyên bố trong câu trả lời chỉ ra được đường đi của nó.
2. **Suy luận nhiều bước** — biến quan sát → rule → cơ chế → PM2.5, và với vận
   chuyển thì thêm nhánh nguồn → hướng gió → PM2.5.
3. **Sinh hình cho khóa luận** — `to_mermaid()` xuất trực tiếp sơ đồ dán vào báo cáo.

Schema cạnh:
    (Variable:blh) -[EVALUATED_BY]-> (Rule:R1) -[TRIGGERS]-> (Mechanism:MECH_LOW_PBLH)
    (Mechanism:MECH_LOW_PBLH) -[INCREASES]-> (Pollutant:PM2.5)
    (Mechanism:MECH_LOW_PBLH) -[SUPPORTED_BY]-> (Paper:10.xxxx)     ← RAG thêm động
"""

from __future__ import annotations

import networkx as nx

from kb import KnowledgeBase
from schemas import Hypothesis

POLLUTANT = "Pollutant:PM2.5"


def node_id(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def build_knowledge_graph(kb: KnowledgeBase) -> nx.DiGraph:
    """Dựng KG tĩnh từ knowledge base. Không phụ thuộc dữ liệu quan sát."""
    graph = nx.DiGraph()
    graph.add_node(POLLUTANT, kind="Pollutant", label="PM2.5")

    for rule in kb.rules.values():
        rule_node = node_id("Rule", rule.id)
        var_node = node_id("Variable", rule.variable)
        graph.add_node(
            var_node, kind="Variable", label=rule.variable, unit=rule.unit
        )
        graph.add_node(
            rule_node,
            kind="Rule",
            label=rule.name,
            threshold=rule.threshold,
            direction=rule.direction,
            source=rule.source,
        )
        graph.add_edge(var_node, rule_node, relation="EVALUATED_BY")

    for mech in kb.mechanisms.values():
        mech_node = node_id("Mechanism", mech.id)
        graph.add_node(
            mech_node,
            kind="Mechanism",
            label=mech.name,
            label_en=mech.name_en,
            category=mech.category,
            prior=mech.confidence_prior,
        )

        for rule_id in mech.triggers.rules:
            graph.add_edge(node_id("Rule", rule_id), mech_node, relation="TRIGGERS", role="primary")
        for rule_id in mech.co_triggers:
            graph.add_edge(node_id("Rule", rule_id), mech_node, relation="TRIGGERS", role="co")

        for canonical, expectation in mech.variables.items():
            var_node = node_id("Variable", canonical)
            if var_node not in graph:
                graph.add_node(var_node, kind="Variable", label=canonical)
            graph.add_edge(
                mech_node, var_node, relation="INVOLVES", expected_shap=expectation.expected_shap
            )

        relation = "INCREASES" if mech.effect == "increase" else "DECREASES"
        graph.add_edge(mech_node, POLLUTANT, relation=relation)

    return graph


def attach_citations(graph: nx.DiGraph, hypotheses: list[Hypothesis]) -> nx.DiGraph:
    """Gắn cạnh SUPPORTED_BY sau khi RAG đã tìm được tài liệu.

    Nhờ bước này, KG trở thành bản ghi provenance đầy đủ của MỘT lần trả lời:
    nhìn vào đồ thị là biết cơ chế nào được kích hoạt bởi rule nào và được bài
    báo nào chống lưng.
    """
    for hypothesis in hypotheses:
        mech_node = node_id("Mechanism", hypothesis.mechanism_id)
        if mech_node not in graph:
            continue
        for citation in hypothesis.citations:
            paper_node = node_id("Paper", citation.doi or citation.evidence_id)
            graph.add_node(
                paper_node,
                kind="Paper",
                label=citation.short(),
                title=citation.title,
                doi=citation.doi,
                year=citation.year,
            )
            graph.add_edge(
                mech_node, paper_node, relation="SUPPORTED_BY", score=round(citation.score, 3)
            )
    return graph


def explain_path(graph: nx.DiGraph, mechanism_id: str) -> list[tuple[str, str, str]]:
    """Chuỗi cạnh giải thích một cơ chế, dạng (nguồn, quan hệ, đích).

    Dùng cho phần "dấu vết suy luận" hiển thị kèm câu trả lời — người dùng bấm
    vào là thấy toàn bộ đường đi, không phải tin lời LLM.
    """
    mech_node = node_id("Mechanism", mechanism_id)
    if mech_node not in graph:
        return []

    path: list[tuple[str, str, str]] = []
    for rule_node in graph.predecessors(mech_node):
        if graph.nodes[rule_node].get("kind") != "Rule":
            continue
        for var_node in graph.predecessors(rule_node):
            path.append((var_node, "EVALUATED_BY", rule_node))
        path.append((rule_node, "TRIGGERS", mech_node))

    for target in graph.successors(mech_node):
        relation = graph.edges[mech_node, target]["relation"]
        if relation in {"INCREASES", "DECREASES", "SUPPORTED_BY"}:
            path.append((mech_node, relation, target))

    return path


def to_mermaid(graph: nx.DiGraph, mechanism_ids: list[str] | None = None) -> str:
    """Xuất sơ đồ Mermaid để dán thẳng vào khóa luận.

    `mechanism_ids=None` vẽ toàn bộ KG (dùng cho chương thiết kế); truyền danh
    sách cụ thể để vẽ dấu vết của một lần trả lời (dùng cho chương kết quả).
    """
    if mechanism_ids is None:
        subgraph = graph
    else:
        keep: set[str] = {POLLUTANT}
        for mech_id in mechanism_ids:
            mech_node = node_id("Mechanism", mech_id)
            if mech_node not in graph:
                continue
            keep.add(mech_node)
            keep.update(graph.predecessors(mech_node))
            keep.update(graph.successors(mech_node))
            for rule_node in list(graph.predecessors(mech_node)):
                keep.update(graph.predecessors(rule_node))
        subgraph = graph.subgraph(keep)

    def safe(name: str) -> str:
        return name.replace(":", "_").replace(".", "_").replace("/", "_").replace("-", "_")

    shape = {
        "Variable": ("([", "])"),
        "Rule": ("{{", "}}"),
        "Mechanism": ("[", "]"),
        "Pollutant": ("((", "))"),
        "Paper": ("[/", "/]"),
    }

    lines = ["flowchart LR"]
    for node, attrs in subgraph.nodes(data=True):
        open_br, close_br = shape.get(attrs.get("kind", ""), ("[", "]"))
        label = str(attrs.get("label", node)).replace('"', "'")
        lines.append(f'    {safe(node)}{open_br}"{label}"{close_br}')
    for src, dst, attrs in subgraph.edges(data=True):
        lines.append(f"    {safe(src)} -->|{attrs.get('relation', '')}| {safe(dst)}")
    return "\n".join(lines)
