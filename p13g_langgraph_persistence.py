"""资料库 13 第 7 节：持久化、Command 和审批重放。

uv run p13g_langgraph_persistence.py  （无需模型，自动启动两个进程，退出后清理临时数据库）
uv run p13g_langgraph_persistence.py start --db /tmp/shop-approval.sqlite
uv run p13g_langgraph_persistence.py approve --db /tmp/shop-approval.sqlite

第二条命令改成 reject 就走拒绝分支；同一 db + thread_id 只能 start 一次。
看什么：真正退出进程后恢复；approve 节点从头重跑；副作用节点与审批节点分开。
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    order_id: str
    amount: int
    answer: str


def approve(state: State) -> Command[Literal["refund", "cancel"]]:
    print("  进入 approve 节点：首次与恢复时都会打印", flush=True)
    decision = interrupt({"order_id": state["order_id"], "amount": state["amount"], "action": "模拟退款"})
    return Command(goto="refund" if decision == "approve" else "cancel")


def refund(state: State) -> dict:
    # 这里只生成文本，不会真的退款。实际付款接口仍需业务幂等键；拆节点不等于 exactly-once。
    return {"answer": f"模拟退款完成：{state['order_id']}，{state['amount']} 元"}


def cancel(state: State) -> dict:
    return {"answer": f"已拒绝 {state['order_id']} 的退款，未执行退款"}


def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("approve", approve)
    builder.add_node("refund", refund)
    builder.add_node("cancel", cancel)
    builder.add_edge(START, "approve")
    # approve 已经用 Command(goto=...) 路由，不再添加一条无条件出边。
    builder.add_edge("refund", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=checkpointer)


def run(action: str, db: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(saver)
        snapshot = graph.get_state(config)
        if action == "start":
            if snapshot.values:
                raise SystemExit("该 thread_id 已存在，请恢复它或用 --thread-id 指定新任务")
            result = graph.invoke({"order_id": "ORD-20261001", "amount": 199}, config, durability="sync")
            print(f"  等待审批：{result['__interrupt__'][0].value}")
            print(f"  下一步：{graph.get_state(config).next}；现在退出进程")
        else:
            if not snapshot.tasks or not any(task.interrupts for task in snapshot.tasks):
                raise SystemExit("此 db + thread_id 没有待审批任务，请先 start，或检查是否已经处理")
            result = graph.invoke(Command(resume=action), config, durability="sync")
            print(f"  {result['answer']}")
            print(f"  下一步：{graph.get_state(config).next}")
            print(f"  保留的历史检查点数：{len(list(graph.get_state_history(config)))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="demo", choices=["demo", "start", "approve", "reject"])
    parser.add_argument("--db", help="手动运行时必须指定的 SQLite 文件路径")
    parser.add_argument("--thread-id", default="refund-1")
    args = parser.parse_args()
    if args.action == "demo":
        with tempfile.TemporaryDirectory(prefix="langgraph-approval-") as directory:
            db = str(Path(directory) / "checkpoint.sqlite")
            for action in ("start", "approve"):
                print(f"\n==== 新进程：{action} ====", flush=True)
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), action, "--db", db], check=True,
                )
    else:
        if not args.db:
            parser.error("start / approve / reject 必须通过 --db 指定同一个数据库")
        run(args.action, args.db, args.thread_id)


if __name__ == "__main__":
    main()

# 试试改：手动 start 后关闭终端，再从新终端 reject；换 thread_id，观察为什么不能恢复旧任务。
# durability="sync" 必须传给 invoke/stream；换成 InMemorySaver 即使 sync 也不会保存到磁盘。
# - 写代码：新增 status 子命令，只读取检查点，显示审批参数与下一节点，不能恢复执行。
