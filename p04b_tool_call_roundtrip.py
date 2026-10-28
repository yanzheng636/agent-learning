# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=3.0", "pydantic>=2.11", "python-dotenv>=1.1,<2", "pydantic-settings>=2.10,<3"]
# ///
"""资料库 04《结构化输出与 Function Calling》：一轮工具调用里，messages 和 tool_calls 是怎么来回的。

uv run p04b_tool_call_roundtrip.py

看什么：
1. 第一次请求：发给模型的是 messages 加工具定义；模型不执行工具，只回一段“想调哪个工具、参数是什么”
2. 程序执行工具：先把模型这一步原样放回 messages，再按 tool_call_id 放回工具结果
3. 第二次请求：模型看到工具结果，写出给顾客的回答
"""

import json

import _shop
from _llm import LLM


def show(title: str, obj) -> None:
    print(f"\n==== {title} ====")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


llm = LLM()
print(llm.describe())

tools = _shop.schemas(["query_order"])
messages = [
    {"role": "system", "content": _shop.SYSTEM_PROMPT},
    {"role": "user", "content": "帮我看下订单 ORD-20261001 现在什么状态"},
]
show("1. 第一次请求带的工具定义（参数的 JSON Schema 由 Pydantic 生成）", tools)

# ---- 1. 第一次请求：模型只“提议”调用工具 ----
reply = llm.chat(messages, tools)
show("1. 模型返回：没有正文，只有 tool_calls", reply.as_message())
print("注意 arguments 是一个 JSON 字符串，不是对象：模型可能写出不合法的 JSON，要由程序校验")

# ---- 2. 程序执行工具，结果按 tool_call_id 放回去 ----
messages.append(reply.as_message())  # 先放模型这一步，否则下一次请求里工具结果找不到对应的调用
for c in reply.tool_calls:
    result = _shop.execute(c.name, c.arguments)  # 当前顾客由程序决定，模型改不了
    messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
show("2. 第二次请求时的 messages（system、user、assistant 的 tool_calls、tool 的结果）", messages[1:])

# ---- 3. 第二次请求：模型根据工具结果回答 ----
reply = llm.chat(messages, tools)
show("3. 模型的最终回答", reply.as_message())
print(f"\n这一轮一共调了 {llm.calls} 次模型：一次决定调工具，一次写回答。")

# 试试改：
# - 把订单号换成 ORD-20261003（别人的订单），看工具返回什么、模型怎么回答
# - 把订单号换成 12345（格式不对），看参数校验的错误信息长什么样
# - 更换 LLM_MODEL，对比不同模型的 tool_calls 和回答
