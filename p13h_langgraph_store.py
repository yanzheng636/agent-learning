# /// script
# requires-python = ">=3.12"
# dependencies = ["langgraph>=1.2,<2"]
# ///
"""资料库 13 第 7 节与资料库 17 的配套扩展：Checkpointer、Store、Runtime Context。

uv run p13h_langgraph_store.py     （不需要模型）

看什么：同一用户换 thread 仍能读取偏好；不同用户的 namespace 隔离；context 由程序注入。
这里两个后端都在内存：Store 的“跨会话”不代表“跨进程持久化”。
"""

from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


class State(TypedDict):
    question: str
    preference: str
    answer: str


def remember(state: State, runtime: Runtime[Context]) -> dict:
    namespace = ("customer_preferences", runtime.context.user_id)
    # 此例把用户明确确认的偏好写入 Store；实际应用还需做来源和有效期管理。
    if state.get("preference"):
        runtime.store.put(namespace, "reply_style", {"value": state["preference"]})
    item = runtime.store.get(namespace, "reply_style")
    style = item.value["value"] if item else "未记录，使用默认风格"
    return {"answer": f"{state['question']} → 当前回复偏好：{style}"}


def build_graph():
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("remember", remember)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


def main() -> None:
    graph = build_graph()
    for user_id, thread_id, preference in [
        ("u1", "conversation-1", "简短中文，先给结论"),
        ("u1", "conversation-2", ""),
        ("u2", "conversation-3", ""),
    ]:
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n==== 用户 {user_id} / 会话 {thread_id} ====")
        print(f"  运行前 checkpoint：{graph.get_state(config).values}")
        result = graph.invoke(
            {"question": "请介绍售后服务", "preference": preference},
            config, context=Context(user_id=user_id),
        )
        print(f"  {result['answer']}")


if __name__ == "__main__":
    main()

# 试试改：最后一轮改成 u1，会读取到什么？重启整个脚本，第一次运行前还存着偏好吗？
# namespace 是业务代码的隔离约定，不是数据库权限系统；服务端仍要鉴权并绑定 thread 所有者。
# - 写代码：给偏好记录添加来源与更新时间，更新时检查写入用户与 namespace 一致。
