# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=3.0",
#     "pydantic>=2.11",
#     "tiktoken>=0.9",
#     "python-dotenv>=1.1,<2",
#     "pydantic-settings>=2.10,<3",
# ]
# ///
"""资料库 24《Agent 为什么翻车》的死循环：三道兜底——步数上限、重复失败检测、token 预算。

uv run p24a_loop_guard.py

看什么：
1. 只有步数上限：模型如果反复调用工具，最多执行到上限
2. 加上重复失败检测：同一工具、同样参数失败两次，就在结果里明确告诉模型“别再试了，去追问顾客”
3. 加上 token 预算：累计输入超过预算就停，把已有进展和缺的信息交代清楚

兜底写在循环里（代码），不是写在提示里请模型自觉：模型负责决策，系统负责守门。
真实模型可能直接追问或成功查到物流，不一定触发重复失败检测或预算上限；观察实际调用记录。
"""

import json

import _shop
from _llm import FALLBACK_REPLY, LLM

STOP_HINT = "同样的调用已经失败了 2 次，不要再重试。请改为向顾客追问缺少的信息。"


def guarded_loop(*, max_steps: int, detect_repeat: bool, token_budget: int | None) -> dict:
    llm = LLM()
    messages = [{"role": "system", "content": _shop.SYSTEM_PROMPT}, {"role": "user", "content": "我的快递到哪了"}]
    tools = _shop.schemas()
    failures: dict[tuple[str, str], int] = {}
    outcome = FALLBACK_REPLY
    for step in range(1, max_steps + 1):
        if token_budget and llm.prompt_tokens > token_budget:  # 调用前检查，所以实际花费会超出预算一次调用的量
            outcome = f"预算用完（已用 {llm.prompt_tokens} token），本轮查询尚未完成，已转人工"
            break
        reply = llm.chat(messages, tools)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            outcome = reply.content or ""
            break
        for c in reply.tool_calls:
            result = json.loads(_shop.execute(c.name, c.arguments))
            if isinstance(result, dict) and "error" in result:
                key = (c.name, c.arguments)
                failures[key] = failures.get(key, 0) + 1
                if detect_repeat and failures[key] >= 2:
                    result["error"] += f" {STOP_HINT}"  # 不是悄悄重试，而是明确告诉模型换做法
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, ensure_ascii=False)})
            print(f"    第 {step} 步 {c.name}({c.arguments}) → {result['error'][:60] if 'error' in result else '成功'}")
    return {"模型调用": llm.calls, "累计输入 token": llm.prompt_tokens, "结果": outcome}


summary = {}
print("==== 1. 只有步数上限（10 步）====")
summary["只有步数上限"] = guarded_loop(max_steps=10, detect_repeat=False, token_budget=None)
print("\n==== 2. 加上重复失败检测 ====")
summary["加重复失败检测"] = guarded_loop(max_steps=10, detect_repeat=True, token_budget=None)
print("\n==== 3. 只有 token 预算（3000）====")
summary["只有 token 预算"] = guarded_loop(max_steps=10, detect_repeat=False, token_budget=3000)

print("\n==== 对比 ====")
for name, r in summary.items():
    print(f"  {name:<10} 调用 {r['模型调用']:>2} 次，累计输入 {r['累计输入 token']:>6} token\n    给顾客：{r['结果']}")
print("三道兜底一起用：重复检测让模型尽早换做法，预算和步数上限兜住检测漏掉的情况")

# 试试改：
# - 把重复检测的阈值从 2 改成 3，看多花了多少
# - 把 STOP_HINT 改成一句含糊的“出错了”，再想想真模型看到后会不会换做法（错误信息要可执行，见资料库 24 篇 2.7 节）
