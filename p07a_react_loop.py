# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=3.0", "pydantic>=2.11", "python-dotenv>=1.1,<2", "pydantic-settings>=2.10,<3"]
# ///
"""资料库 07《Agent 循环》：手写一个 ReAct 循环，看模型、工具、观察怎么一圈圈转起来。

uv run p07a_react_loop.py

看什么：
1. 每一步：模型决定动作 → 程序执行工具 → 结果回填 → 模型再决定，直到模型不再要求调用工具
2. 终止条件：模型直接回答，或者走到步数上限
3. 到了上限还没答完，就给一句固定的兜底回复，不把半截的工具调用丢给用户

这个循环和参考项目 shop-assistant 的 assistant/agent.py 是同一个思路，那边换成了流式输出和并发执行工具。
"""

import _shop
from _llm import FALLBACK_REPLY, LLM

MAX_STEPS = 5


def react(llm: LLM, question: str, max_steps: int = MAX_STEPS) -> str:
    messages = [{"role": "system", "content": _shop.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    tools = _shop.schemas()
    for step in range(1, max_steps + 1):
        reply = llm.chat(messages, tools)  # 1. 模型决定下一步
        messages.append(reply.as_message())
        if not reply.tool_calls:  # 2. 不再调用工具，说明它给出了最终回答
            print(f"  第 {step} 步  回答：{reply.content}")
            return reply.content or ""
        for c in reply.tool_calls:  # 3. 执行模型要求的工具
            result = _shop.execute(c.name, c.arguments)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})  # 4. 观察回填
            print(f"  第 {step} 步  动作：{c.name}({c.arguments})\n         观察：{result[:90]}")
    print(f"  走到了 {max_steps} 步上限，模型还在要求调用工具，改用兜底回复")
    return FALLBACK_REPLY


print("==== 1. 正常的一轮：列订单 → 查物流 → 回答 ====")
llm = LLM()
print(llm.describe())
answer = react(llm, "我上周买的耳机到哪了？")
print(f"  共调用模型 {llm.calls} 次")

print("\n==== 2. 把步数上限降到 1：观察多步任务如何触发兜底 ====")
print(f"  给用户的回复：{react(LLM(), '我上周买的耳机到哪了？', max_steps=1)}")

# 试试改：
# - 把问题换成“电饭煲什么时候发货”，看模型会怎么选工具
# - 把 MAX_STEPS 改成 2，看第 1 段会不会触发兜底
# - 更换模型，对比完成同一个任务所需的步数
# - 写代码：接入 p24a 的 FailureGuard；同一查询失败两次就退出，并打印触发原因。
