"""资料库 13 第 3、5～7 节：显式 ReAct 图与流式事件配套扩展。

uv run p13e_langgraph_react.py

看什么：MessagesState、bind_tools、ToolNode、条件边、updates / messages 双路事件。
和 p07a 的手写循环、p13d 的 create_agent 对照；三者都是真实模型调用。
"""

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import _shop
from _langchain import SHOP_TOOLS, make_model


def build_graph():
    model = make_model().bind_tools(SHOP_TOOLS)

    def call_model(state: MessagesState) -> dict:
        reply = model.invoke([SystemMessage(content=_shop.SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}  # 只返回增量，不重复返回整段历史。

    # MessagesState 的 messages 使用 add_messages：不同 id 追加，同 id 更新。
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(SHOP_TOOLS))  # 执行工具，并生成带 tool_call_id 的 ToolMessage。
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)  # 有 tool_calls → tools，否则 → END。
    builder.add_edge("tools", "model")
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "react-u1"}, "recursion_limit": 20}
    print("==== 1. 模型 token 与节点增量是不同事件 ====")
    for mode, chunk in graph.stream(
        {"messages": [("user", "帮我查订单 ORD-20261001 的商品、金额和物流。")]},
        config,
        stream_mode=["updates", "messages"],
        durability="sync",  # 等检查点写完再走下一步；内存后端仍不耐进程退出。
    ):
        if mode == "messages":
            token, metadata = chunk
            if token.content:
                print(f"[token/{metadata['langgraph_node']}] {token.content}")
        else:
            for node, delta in chunk.items():
                print(f"[完成/{node}]")
                for message in delta.get("messages", []):
                    if message.type == "tool":
                        print(f"  工具结果 id={message.tool_call_id}：{message.content}")
                    for call in getattr(message, "tool_calls", []):
                        print(f"  请求 {call['name']}({call['args']})，id={call['id']}")

    print("\n==== 2. 从检查点查看最终消息与待执行节点 ====")
    snapshot = graph.get_state(config)
    print(snapshot.values["messages"][-1].content)
    print(f"  下一步：{snapshot.next}（空元组表示这轮已完成）")


if __name__ == "__main__":
    main()

# 试试改：stream_mode="values" 观察完整状态；改为只用 updates 看清循环次数。
# tools_condition 只决定路由；bind_tools 只告诉模型工具 Schema，真正执行的是 ToolNode。
# - 写代码：统计每次运行中 model 与 tools 节点的完成次数，单独统计工具调用次数。
