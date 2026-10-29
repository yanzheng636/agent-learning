"""资料库 13 第 6 节：super-step、reducer；用 Send 动态并行的配套扩展。

uv run p13f_langgraph_map_reduce.py     （不需要模型）

看什么：先亲眼看到并发写冲突，再用 Send 动态派发、reducer 汇合、defer=True 汇总。
分析节点用确定性代码提取政策；替换成模型也不改变图的调度机制。
"""

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class BadState(TypedDict):
    findings: list[str]


class State(TypedDict):
    docs: list[str]
    findings: Annotated[list[dict], operator.add]
    report: str


class BranchInput(TypedDict):
    index: int
    doc: str
    before_count: int


def fan_out(state: State) -> list[Send] | str:
    # 空列表没有 worker，也必须能走到汇总节点。
    return [
        Send("analyze", {"index": i, "doc": doc, "before_count": len(state["findings"])})
        for i, doc in enumerate(state["docs"])
    ] or "summarize"


def analyze(state: BranchInput) -> dict:
    # 每个分支收到 Send 的私有输入，不会读到同一轮其他分支刚返回的更新。
    return {"findings": [{"index": state["index"], "text": state["doc"], "before_count": state["before_count"]}]}


def summarize(state: State) -> dict:
    # reducer 负责收集，不把业务顺序交给并行完成顺序；按显式 index 排序。
    ordered = sorted(state["findings"], key=lambda item: item["index"])
    return {"report": "\n".join(f"  {item['index'] + 1}. {item['text']}" for item in ordered) or "  没有政策待分析"}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("analyze", analyze)
    builder.add_node("summarize", summarize, defer=True)
    builder.add_conditional_edges(START, fan_out, ["analyze", "summarize"])
    builder.add_edge("analyze", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def main() -> None:
    print("==== 1. 同一轮写同一个无 reducer 字段，框架拒绝猜合并方式 ====")
    bad = StateGraph(BadState)
    bad.add_node("a", lambda state: {"findings": ["分支 A"]})
    bad.add_node("b", lambda state: {"findings": ["分支 B"]})
    for node in ("a", "b"):
        bad.add_edge(START, node)
        bad.add_edge(node, END)
    try:
        bad.compile().invoke({"findings": []})
    except InvalidUpdateError:
        print("  捕获预期的 InvalidUpdateError：并发写 findings 时缺少 reducer")

    print("\n==== 2. Send 派发三路，reducer 合并，最后只汇总一次 ====")
    graph = build_graph()
    inputs = {"docs": ["签收后 7 天内可无理由退货", "质量问题运费由小店承担", "耳机保修半年"], "findings": []}
    for update in graph.stream(inputs, stream_mode="updates"):
        print(f"  增量：{update}")

    print("\n==== 3. 空输入也能结束 ====")
    print(graph.invoke({"docs": [], "findings": []})["report"])


if __name__ == "__main__":
    main()

# 试试改：增删 docs，观察 analyze 执行次数；给某一路加延时，观察 summarize 何时执行。
# 不要让 analyze 原地修改输入列表；只返回本分支产生的增量。
# - 写代码：在汇总前按文档 index 检查重复和缺失结果；空文档集要有明确返回值。
