"""资料库 22《多 Agent》：把一个子 Agent 包装成工具（Agent-as-Tool），看上下文隔离换来了什么、花了什么。

uv run p22a_agent_as_tool.py

看什么：
1. 单 Agent：查所有订单、查物流、查政策，全部工具结果都堆在主上下文里
2. Agent-as-Tool：主 Agent 把“查订单”整件事交给子 Agent；子 Agent 有自己的上下文和工具，只交回一段紧凑的结论
3. 对比：主上下文变小，模型调用次数变多；总的输入 token 是多是少，取决于每个 Agent 带了多少工具定义和上下文。
   这个小例子里子 Agent 只带 3 个工具，总量也可能更少；Anthropic 在真实任务里观察到的是通常多 3 到 10 倍，
   因为上下文要在多个 Agent 里重复、要传任务说明和汇总结果（资料库 22 篇）
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import _shop
from _llm import LLM, count_tokens, run_agent

QUESTION = "我所有订单都什么情况？顺便说下无理由退货运费谁出"


# ---- 子 Agent，包装成一个工具 ----
SUB_TOOLS = _shop.schemas(["list_my_orders", "query_order", "track_shipment"])
SUB_PROMPT = '你负责查订单。查完只输出 JSON：{"结论": "一句话", "证据": ["用过的工具"], "状态": "完成/部分完成/失败"}'
sub_llm = LLM()


def research_orders(task: str) -> str:
    sub_messages = [{"role": "system", "content": SUB_PROMPT}, {"role": "user", "content": task}]  # 全新的上下文
    result = run_agent(sub_llm, sub_messages, SUB_TOOLS, _shop.execute, label="    子 Agent ")
    own, back = count_tokens(sub_messages), count_tokens(result)
    print(f"    子 Agent 自己的上下文有 {own} token，交回给主 Agent 的只有 {back} token")
    return result


class ResearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, description="要查什么")


RESEARCH_TOOL = {"type": "function", "function": {
    "name": "research_orders",
    "description": "交给订单子 Agent：它会自己查订单和物流，返回一句话结论、用过的工具和完成状态。",
    "parameters": ResearchArgs.model_json_schema(),
}}


def main_execute(name: str, arguments: str) -> str:
    if name != "research_orders":
        return _shop.execute(name, arguments)
    try:
        args = ResearchArgs.model_validate_json(arguments or "{}")
    except ValidationError as exc:
        return _shop.args_error(exc)  # 参数不合法就回填错误，不启动子 Agent
    return research_orders(args.task)


# ---- 1. 单 Agent ----
print("==== 1. 单 Agent：所有工具结果都在主上下文里 ====")
single = LLM()
print(single.describe())
msgs1 = [{"role": "system", "content": _shop.SYSTEM_PROMPT}, {"role": "user", "content": QUESTION}]
answer1 = run_agent(single, msgs1, _shop.schemas(), _shop.execute)
print(f"  回答：{answer1}")

# ---- 2. Agent-as-Tool ----
print("\n==== 2. Agent-as-Tool：主 Agent 只拿到子 Agent 的结论 ====")
main = LLM()
msgs2 = [{"role": "system", "content": _shop.SYSTEM_PROMPT}, {"role": "user", "content": QUESTION}]
answer2 = run_agent(main, msgs2, _shop.schemas(["search_knowledge"]) + [RESEARCH_TOOL], main_execute)
print(f"  回答：{answer2}")

# ---- 3. 对比 ----
print("\n==== 3. 对比 ====")
print(f"  {'':<14}{'主上下文最终大小':>12}{'模型调用次数':>10}{'累计输入 token':>14}")
print(f"  {'单 Agent':<14}{count_tokens(msgs1):>12}{single.calls:>14}{single.prompt_tokens:>16}")
print(f"  {'Agent-as-Tool':<14}{count_tokens(msgs2):>12}{main.calls + sub_llm.calls:>14}"
      f"{main.prompt_tokens + sub_llm.prompt_tokens:>16}")
single_total, multi_total = single.prompt_tokens, main.prompt_tokens + sub_llm.prompt_tokens
print(f"  同样的查询，主上下文小了 {100 - count_tokens(msgs2) * 100 // count_tokens(msgs1)}%；"
      f"模型调用多了 {main.calls + sub_llm.calls - single.calls} 次，"
      f"累计输入 token {'多' if multi_total > single_total else '少'}了 "
      f"{abs(multi_total - single_total) * 100 // single_total}%")
print("  子 Agent 的中间过程主 Agent 看不到，跑偏了也没法中途纠正；每多一层就多一轮完整的模型循环，延迟也跟着加")
if multi_total <= single_total:
    print("  token 少了是因为子 Agent 只带 3 个工具定义、主 Agent 只带 2 个；")
    print("  专门化能省工具定义，但任务一复杂，重复的上下文和汇总就会把它吃掉")
print("  这个例子里订单只有两笔，单 Agent 完全够用。值得拆的是工具结果很大、和主任务无关、或者能并行的场景")

# 试试改：
# - 在 _shop.ORDERS 里给 u1 多加几笔订单，看两种做法的主上下文差距怎么变
# - 修改 SUB_PROMPT，让子 Agent 返回原始数据而不是 JSON 结论，看主上下文大小怎么变
# - 写代码：给子 Agent 返回值定义 Pydantic Schema，校验来源和订单号后再交给主 Agent。
