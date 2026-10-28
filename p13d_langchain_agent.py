# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain>=1.4,<2",
#     "langchain-openai>=1.7,<2",
#     "langgraph>=1.2,<2",
#     "python-dotenv>=1.1,<2",
#     "pydantic-settings>=2.10,<3",
# ]
# ///
"""资料库 13 · LangChain 家族配套 · LangChain Agent：create_agent、工具调用、会话记忆。

uv run p13d_langchain_agent.py

看什么：和 p07a 手写循环对照；框架执行模型 → 工具 → 模型，按 thread_id 保存消息。
模型要支持 tool calling。InMemorySaver 只在当前进程有效，不是长期记忆。
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import _shop
from _langchain import SHOP_TOOLS, make_model


def main() -> None:
    agent = create_agent(
        model=make_model(),
        tools=SHOP_TOOLS,
        system_prompt=_shop.SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "customer-u1"}, "recursion_limit": 20}

    print("==== 1. 模型选工具，框架执行并回填结果 ====")
    for update in agent.stream(
        {"messages": [{"role": "user", "content": "帮我查订单 ORD-20261001 的物流。"}]},
        config, stream_mode="updates",
    ):
        for node, delta in update.items():
            print(f"  节点：{node}")
            for message in delta.get("messages", []):
                print(f"    {message.type}：{message.content}")
                for call in getattr(message, "tool_calls", []):
                    print(f"    请求工具：{call['name']}({call['args']})，id={call['id']}")

    print("\n==== 2. 同一个 thread_id：只传新问题，框架加载已有消息 ====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "这笔订单买的什么，花了多少钱？"}]}, config,
    )
    print(result["messages"][-1].content)
    print(f"  当前会话消息数：{len(result['messages'])}")

    print("\n==== 3. 新 thread_id 没有之前的会话 ====")
    fresh = agent.get_state({"configurable": {"thread_id": "new-conversation"}})
    print(f"  新会话状态：{fresh.values}")


if __name__ == "__main__":
    main()

# 试试改：第二轮换一个 thread_id，观察模型是否会追问订单号或重新查订单。
# recursion_limit 限制图的执行步数，不等同于模型调用次数，也不是超时或 token 预算。
