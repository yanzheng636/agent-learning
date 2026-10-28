# /// script
# requires-python = ">=3.12"
# dependencies = ["langgraph>=1.2,<2"]
# ///
"""资料库 13 第 7 节检查点的配套扩展：历史检查点与分支重放。

uv run p13j_langgraph_time_travel.py     （不需要模型）

看什么：普通配送报价 → 回到报价前 → 改成加急 → 生成新分支，原历史仍可查看。
关键：update_state 返回新 checkpoint config；必须拿它继续，不能拿旧 past.config 恢复。
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    shipping: str
    fee: int
    answer: str


def prepare(state: State) -> dict:
    return {"shipping": state["shipping"]}


def quote(state: State) -> dict:
    return {"fee": 20 if state["shipping"] == "加急" else 8}


def finish(state: State) -> dict:
    return {"answer": f"{state['shipping']}配送，演示运费 {state['fee']} 元（未创建订单）"}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("prepare", prepare)
    builder.add_node("quote", quote)
    builder.add_node("finish", finish)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "quote")
    builder.add_edge("quote", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "shipping-demo"}}
    print("==== 1. 运行原始分支 ====")
    original = graph.invoke({"shipping": "普通"}, config)
    original_config = graph.get_state(config).config
    print(original["answer"])

    print("\n==== 2. 按下一步节点定位历史点，不依赖 history[3] 之类的固定下标 ====")
    past = next(snapshot for snapshot in graph.get_state_history(config) if snapshot.next == ("quote",))
    print(f"  历史状态：{past.values}，下一步：{past.next}")
    branch_config = graph.update_state(past.config, {"shipping": "加急"})
    branch = graph.invoke(None, branch_config)
    print(f"  新分支：{branch['answer']}")

    print("\n==== 3. 原始 checkpoint 仍保留普通配送报价 ====")
    print(graph.get_state(original_config).values["answer"])


if __name__ == "__main__":
    main()

# 试试改：改写历史 fee 后重新执行 quote，观察为什么报价节点会覆盖手动填入的金额。
# 时间旅行不撤销外部世界的副作用；本例只有纯计算，适合安全地比较分支。
# - 写代码：增加第三种配送方式并创建第二条历史分支，断言原始报价和两个分支互不覆盖。
