"""回放样本最小覆盖（minimal replay-sample coverage）选择引擎。

在样本回放诊断（``replay.py``）的基础上回答一个更进一步的问题：**提交的
这一批文档里，最少保留哪几篇，就能把报告中的 issue 全部"实锤"一遍？**

引擎复用回放的旧 / 新双版校验与 issue 归因逻辑，为每篇文档算出它能覆盖
的报告 issue 集合（按 ``direction + path + code + 规则主体`` 身份归并），
随后用**确定性贪心集合覆盖**挑出尽可能少的代表文档：

* 每轮选中"新增覆盖 issue 数"最多的文档；
* 并列时按文档下标升序（先提交者优先），全程无随机源，同一输入必得同一
  输出；
* 直到目标方向内的 issue 被覆盖完，或剩余文档都无法再贡献新覆盖为止。

方向筛选
========
``directions`` 限定参与覆盖的兼容方向（缺省 ``backward`` 与 ``forward``
都参与）：只有被归类为该方向的文档才参与选择，也只有该方向的报告
issue 计入覆盖目标。``valid`` / ``invalid_both`` 文档不覆盖任何 issue，
但仍计入 ``counts`` 原始分类统计。

引擎为纯函数，不读取数据库、不修改入参、不做任何持久化。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

from app.compat import make_validator
from app.replay import (
    _attribute,
    _build_record,
    _collect_leaves,
    _find_false_pointers,
    _index_issues,
    _is_false_leaf,
    _issue_subject,
)

# 单次覆盖选择允许的最大文档数（与 schemas.ReplayCoverageRequest 保持一致）。
MAX_COVERAGE_DOCUMENTS = 5000

# 方向的规范输出顺序。
_DIRECTION_ORDER = ("backward", "forward")


def _document_issue_keys(document: Any, direction: str, failing_errors: list,
                         failing_schema: Any, issue_index: dict) -> set:
    """算出单篇文档在指定方向上可归因到的报告 issue 身份集合。

    与回放一致：先展平 oneOf/anyOf 错误树，出现 false 子 schema 叶子时在
    失败侧 schema 上预扫描定位，再逐叶子归因；同一文档对同一条 issue 只
    记一次（集合去重）。
    """
    leaves: list[Any] = []
    for top_error in failing_errors:
        leaves.extend(_collect_leaves(top_error))

    false_map: dict[int, list[str]] = {}
    if any(_is_false_leaf(leaf) for leaf in leaves):
        false_map = _find_false_pointers(failing_schema, document)

    keys: set[tuple[str, str, str, str]] = set()
    for leaf in leaves:
        is_false = _is_false_leaf(leaf)
        false_pointers = (
            list(false_map.get(id(leaf.instance), [])) if is_false else None
        )
        record = _build_record(leaf, false_pointers)
        issue = _attribute(record, direction, issue_index)
        if issue is not None:
            keys.add((direction, issue["path"], issue["code"],
                      _issue_subject(issue)))
    return keys


def _greedy_set_cover(doc_covers: dict[int, set], universe: set) -> list[int]:
    """确定性贪心集合覆盖：每轮选新增覆盖最多的文档，并列取下标较小者。"""
    uncovered = set(universe)
    selected: list[int] = []
    # 候选按下标升序扫描，严格大于才替换最优 -> 并列时下标较小者胜出
    remaining = sorted(doc_covers)
    while uncovered:
        best_index: int | None = None
        best_gain = 0
        for index in remaining:
            gain = len(doc_covers[index] & uncovered)
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        selected.append(best_index)
        uncovered -= doc_covers[best_index]
        remaining.remove(best_index)
    return selected


def select_covering_documents(old_schema: Any, new_schema: Any, report: dict,
                              documents: list[Any],
                              directions: Iterable[str] | None = None) -> dict:
    """从提交文档中贪心选出能覆盖报告 issue 的最少代表文档（纯函数）。

    ``documents`` 由调用方保证为 1..MAX_COVERAGE_DOCUMENTS 个任意 JSON 值；
    ``directions`` 为参与覆盖的方向（None 表示两个方向都参与）。
    """
    requested = set(directions) if directions is not None else set(_DIRECTION_ORDER)
    effective = [name for name in _DIRECTION_ORDER if name in requested]

    old_validator = make_validator(old_schema)
    new_validator = make_validator(new_schema)
    issue_index = _index_issues(report)

    counts = {"valid": 0, "backward": 0, "forward": 0, "invalid_both": 0}
    # 文档下标 -> 该文档可覆盖的报告 issue 身份集合（仅目标方向）
    doc_covers: dict[int, set] = {}

    for index, document in enumerate(documents):
        old_errors = list(old_validator.iter_errors(document))
        new_errors = list(new_validator.iter_errors(document))
        old_ok, new_ok = not old_errors, not new_errors

        if old_ok and new_ok:
            counts["valid"] += 1
            continue
        if not old_ok and not new_ok:
            counts["invalid_both"] += 1
            continue

        if old_ok:
            direction, failing_errors, failing_schema = (
                "backward", new_errors, new_schema)
        else:
            direction, failing_errors, failing_schema = (
                "forward", old_errors, old_schema)
        counts[direction] += 1
        if direction not in requested:
            continue

        keys = _document_issue_keys(
            document, direction, failing_errors, failing_schema, issue_index)
        if keys:
            doc_covers[index] = keys

    # 覆盖目标：目标方向内的全部报告 issue，按报告内顺序去重（身份同回放）
    universe: "OrderedDict[tuple[str, str, str, str], dict]" = OrderedDict()
    for issue in report.get("issues", []):
        if issue["direction"] not in requested:
            continue
        key = (issue["direction"], issue["path"], issue["code"],
               _issue_subject(issue))
        universe.setdefault(key, issue)

    selected = _greedy_set_cover(doc_covers, set(universe))

    # 每条 issue 首次被哪篇选中文档覆盖（按选中顺序记录）
    covered_by: dict[tuple[str, str, str, str], int] = {}
    for index in selected:
        for key in doc_covers[index]:
            covered_by.setdefault(key, index)

    covered_issues: list[dict] = []
    uncovered_issues: list[dict] = []
    for key, issue in universe.items():
        entry = {
            "direction": issue["direction"],
            "path": issue["path"],
            "code": issue["code"],
            "severity": issue["severity"],
        }
        if key in covered_by:
            covered_issues.append({**entry, "covered_by": covered_by[key]})
        else:
            uncovered_issues.append(entry)

    return {
        "total_documents": len(documents),
        "directions": effective,
        "counts": counts,
        "selected_indices": selected,
        "covered_issues": covered_issues,
        "uncovered_issues": uncovered_issues,
    }
