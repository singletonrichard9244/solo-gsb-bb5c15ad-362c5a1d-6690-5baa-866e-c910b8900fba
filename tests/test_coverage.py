"""回放样本最小覆盖引擎单元测试。

覆盖：逐文档 issue 覆盖集合构建、确定性贪心选择（最大新增覆盖优先、
并列取文档下标较小者）、backward/forward 方向筛选、原始四向分类统计、
未覆盖 issue 输出、同 path/code 多主体 issue 区分，以及无法归因文档
不参与覆盖。
"""
from app import compat, coverage

# 报告 issue（全部 backward，按报告内顺序）：
#   # required_added(s) / required_added(x) / required_added(y)
#   #/properties/s enum_value_removed
OLD = {"type": "object", "properties": {"s": {"enum": ["a", "b", "c"]}}}
NEW = {
    "type": "object",
    "properties": {"s": {"enum": ["a", "b"]}, "x": {}, "y": {}},
    "required": ["s", "x", "y"],
}

# 双向场景：backward enum_value_removed(#/properties/s)，
# forward required_added(email, #)
OLD_B = {
    "type": "object",
    "required": ["email"],
    "properties": {"email": {"type": "string"}, "s": {"enum": ["a", "b"]}},
}
NEW_B = {
    "type": "object",
    "properties": {"email": {"type": "string"}, "s": {"enum": ["a"]}},
}
DOCS_B = [
    {"email": "e", "s": "b"},   # 0: backward -> {enum_value_removed}
    {"s": "a"},                 # 1: forward  -> {required_added(email)}
    {"email": "e", "s": "a"},   # 2: valid
]


def _report(old=OLD, new=NEW):
    return compat.analyze(old, new)


# ---------------------------------------------------------------------------
# 贪心选择与并列打破
# ---------------------------------------------------------------------------


def test_greedy_prefers_largest_new_coverage():
    report = _report()
    result = coverage.select_covering_documents(OLD, NEW, report, [
        {"s": "c", "x": 1, "y": 2},   # 0 -> {enum}
        {"s": "a", "y": 2},           # 1 -> {required x}
        {"s": "a"},                   # 2 -> {required x, required y}
        {"x": 1, "y": 2},             # 3 -> {required s}
        {"s": "a", "x": 1, "y": 2},   # 4 -> valid，不覆盖任何 issue
        42,                           # 5 -> invalid_both，不覆盖任何 issue
    ])

    # 先选新增覆盖 2 条的 doc2；doc0/doc3 并列 1 条时按下标选 doc0，最后 doc3
    assert result["selected_indices"] == [2, 0, 3]
    assert result["total_documents"] == 6
    assert result["counts"] == {
        "valid": 1, "backward": 4, "forward": 0, "invalid_both": 1,
    }
    # covered_issues 按报告内 issue 顺序：required s/x/y、enum
    assert [i["covered_by"] for i in result["covered_issues"]] == [3, 2, 2, 0]
    assert [i["code"] for i in result["covered_issues"]] == [
        "required_added", "required_added", "required_added",
        "enum_value_removed",
    ]
    assert result["uncovered_issues"] == []


def test_tie_broken_by_document_index():
    report = _report()
    result = coverage.select_covering_documents(OLD, NEW, report, [
        {"s": "c", "x": 1, "y": 2},   # 0 -> {enum}
        {"s": "c", "x": 1, "y": 2},   # 1 -> {enum}（与 doc0 完全相同）
        {"x": 1, "y": 2},             # 2 -> {required s}
    ])

    # 首轮三者新增覆盖都是 1，按下标选 doc0；doc1 不再贡献新覆盖被跳过
    assert result["selected_indices"] == [0, 2]
    # required x / y 无任何文档覆盖
    assert [i["code"] for i in result["uncovered_issues"]] == [
        "required_added", "required_added",
    ]
    assert all(i["direction"] == "backward" for i in result["uncovered_issues"])


def test_deterministic_across_runs():
    report = _report()
    documents = [
        {"s": "c", "x": 1, "y": 2},
        {"s": "a"},
        {"x": 1, "y": 2},
        {"s": "b", "x": 1},
    ]
    first = coverage.select_covering_documents(OLD, NEW, report, documents)
    second = coverage.select_covering_documents(OLD, NEW, report, documents)
    assert first == second


def test_subject_distinguishes_same_path_code_issues():
    # 三条 required_added 同在 #、code 相同，仅靠规则主体（缺失字段名）区分
    report = _report()
    result = coverage.select_covering_documents(OLD, NEW, report, [
        {"s": "a", "x": 1, "y": 2},   # 0: valid
        {"s": "a", "y": 2},           # 1: 只缺 x -> 只覆盖 required_added(x)
    ])
    assert result["selected_indices"] == [1]
    assert len(result["covered_issues"]) == 1
    assert result["covered_issues"][0]["code"] == "required_added"
    assert result["covered_issues"][0]["covered_by"] == 1
    # required s / y 与 enum 仍未覆盖
    assert len(result["uncovered_issues"]) == 3


# ---------------------------------------------------------------------------
# 方向筛选
# ---------------------------------------------------------------------------


def test_default_directions_cover_both():
    report = compat.analyze(OLD_B, NEW_B)
    result = coverage.select_covering_documents(OLD_B, NEW_B, report, DOCS_B)
    assert result["directions"] == ["backward", "forward"]
    assert result["selected_indices"] == [0, 1]
    assert [i["direction"] for i in result["covered_issues"]] == [
        "backward", "forward",
    ]
    assert result["uncovered_issues"] == []


def test_direction_filter_limits_universe_and_candidates():
    report = compat.analyze(OLD_B, NEW_B)
    result = coverage.select_covering_documents(
        OLD_B, NEW_B, report, DOCS_B, ["forward"])
    assert result["directions"] == ["forward"]
    # 只有 forward 文档参与选择，只有 forward issue 计入覆盖目标
    assert result["selected_indices"] == [1]
    assert [i["direction"] for i in result["covered_issues"]] == ["forward"]
    assert result["uncovered_issues"] == []
    # 原始分类统计不受方向筛选影响
    assert result["counts"] == {
        "valid": 1, "backward": 1, "forward": 1, "invalid_both": 0,
    }


def test_backward_only_excludes_forward_issue_from_universe():
    report = compat.analyze(OLD_B, NEW_B)
    result = coverage.select_covering_documents(
        OLD_B, NEW_B, report, DOCS_B, ["backward"])
    assert result["directions"] == ["backward"]
    assert result["selected_indices"] == [0]
    # forward issue 不在覆盖目标内，不会出现在 uncovered 中
    assert result["uncovered_issues"] == []
    assert all(i["direction"] == "backward" for i in result["covered_issues"])


# ---------------------------------------------------------------------------
# 未覆盖与无法归因
# ---------------------------------------------------------------------------


def test_uncovered_issues_listed_when_no_document_matches():
    report = _report()
    result = coverage.select_covering_documents(OLD, NEW, report, [
        {"s": "a", "x": 1, "y": 2},   # valid
        42,                           # invalid_both
    ])
    assert result["selected_indices"] == []
    assert result["covered_issues"] == []
    assert len(result["uncovered_issues"]) == 4
    assert {i["severity"] for i in result["uncovered_issues"]} == {"error"}
    assert result["counts"] == {
        "valid": 1, "backward": 0, "forward": 0, "invalid_both": 1,
    }


def test_unattributable_failure_covers_nothing():
    # dependentRequired 不在分析器覆盖的关键字内：文档被新版拒绝（backward）
    # 但失败无法归因到任何报告 issue，因而不参与覆盖
    old = {"type": "object"}
    new = {"type": "object", "dependentRequired": {"a": ["b"]}}
    report = compat.analyze(old, new)
    assert report["issues"] == []

    result = coverage.select_covering_documents(old, new, report, [{"a": 1}])
    assert result["counts"]["backward"] == 1
    assert result["selected_indices"] == []
    assert result["covered_issues"] == []
    assert result["uncovered_issues"] == []
