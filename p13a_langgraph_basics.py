# /// script
# requires-python = ">=3.12"
# dependencies = ["langgraph>=1.2"]
# ///
"""资料库 13《LangGraph》：State、节点和边、条件分支、检查点、interrupt 人工介入，一个文件看全。

uv run p13a_langgraph_basics.py      （不需要模型：节点都是普通函数，真实项目里节点里面调模型）

看什么：
1. State：图里流动的数据；带“合并规则”的字段（这里的 log）每个节点追加，不会互相覆盖
2. 节点和边：分类 → 条件分支 → 查订单或答政策；退款分支要先过人工审批
3. 检查点：每走一步自动存一份状态，按 thread_id 区分会话
4. interrupt：图停在审批节点；恢复会重跑整个审批节点。跨进程恢复见 p13g 的 SQLite 示例
5. 历史：每一步的状态都还在，可以回看
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    question: str
    intent: str
    answer: str
    log: Annotated[list[str], operator.add]  # 合并规则：各节点返回的 log 追加到一起


def classify(state: State) -> dict:
    q = state["question"]
    intent = "退款" if "退" in q and "政策" not in q else "订单" if "订单" in q or "到哪" in q else "政策"
    return {"intent": intent, "log": [f"classify → {intent}"]}


def lookup_order(state: State) -> dict:
    return {"answer": "订单 ORD-20261001 顺丰派送中", "log": ["lookup_order"]}


def answer_policy(state: State) -> dict:
    return {"answer": "签收后 7 天内可无理由退货", "log": ["answer_policy"]}


def approve_refund(state: State) -> dict:
    print("  进入 approve_refund（恢复时会再次打印；这里不能放扣款等不可重复的副作用）")
    decision = interrupt({"待审批": "给 ORD-20261001 退款 199 元", "问": "批准吗？"})  # 图在这里停下
    answer = "已提交退款" if decision == "批准" else "已拒绝退款，转人工"
    return {"answer": answer, "log": [f"approve_refund → {decision}"]}


def route(state: State) -> str:
    return {"退款": "approve_refund", "订单": "lookup_order"}.get(state["intent"], "answer_policy")


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("lookup_order", lookup_order)
graph.add_node("answer_policy", answer_policy)
graph.add_node("approve_refund", approve_refund)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route, ["lookup_order", "answer_policy", "approve_refund"])
for node in ["lookup_order", "answer_policy", "approve_refund"]:
    graph.add_edge(node, END)
app = graph.compile(checkpointer=InMemorySaver())  # 生产环境换成 Postgres 之类的持久化检查点

print("==== 1. 普通问题：一口气走完 ====")
for i, q in enumerate(["我的订单到哪了", "退货政策是什么"]):
    out = app.invoke({"question": q, "log": []}, {"configurable": {"thread_id": f"t{i}"}})
    print(f"  {q} → {out['answer']}  路径：{out['log']}")

print("\n==== 2. 退款：停在审批节点 ====")
cfg = {"configurable": {"thread_id": "refund-1"}}
out = app.invoke({"question": "耳机坏了帮我退款", "log": []}, cfg)
snapshot = app.get_state(cfg)
print(f"  图停在：{snapshot.next}，等待：{out['__interrupt__'][0].value}")
print("  状态在检查点里。换成持久化的检查点后，此时进程可以退出，审批几小时后才来也能接着跑")

print("\n==== 3. 人批准后，从断点继续 ====")
out = app.invoke(Command(resume="批准"), cfg)  # 同一个 thread_id，重跑 approve_refund，interrupt 返回批准
print(f"  结果：{out['answer']}  路径：{out['log']}")

print("\n==== 4. 回看这个会话每一步的状态 ====")
for h in app.get_state_history(cfg):
    print(f"  下一步 {h.next or ('结束',)}，当时的 log：{h.values.get('log')}")

# 试试改：
# - 第 3 段改成 Command(resume="拒绝")，看走向
# - 给 State 加一个 attempts: Annotated[int, operator.add] 字段，在每个节点 +1，看合并规则怎么起作用
