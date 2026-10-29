"""资料库 07《Agent 循环》的 Plan-and-Execute：先让模型列出全部步骤和依赖，再由程序并发执行。

uv run p07c_plan_execute.py

看什么：
1. 规划：模型一次输出计划（JSON），每一步写明调哪个工具、依赖哪几步
2. 执行：程序按依赖分批，同一批里互不依赖的步骤并发跑；全程不再问模型
3. 汇总：最后再调一次模型写回答
4. 和 ReAct 对比：ReAct 每一步都要等模型决定，这里模型只调两次

工具加了模拟的网络延迟（0.5 秒），好让并发的效果看得出来。
"""

import asyncio
import json
import time

from pydantic import BaseModel, ValidationError

import _shop
from _llm import LLM

LATENCY = 0.5

PLANNER_PROMPT = """\
把顾客的问题拆成工具调用计划，只输出 JSON：{"steps": [{"id": "s1", "tool": "...", "args": {...}, "depends_on": []}]}
可用工具：search_products(keyword)、track_shipment(order_id)、query_order(order_id)、list_my_orders()。
互不依赖的步骤 depends_on 留空，它们会被并发执行。"""


class Step(BaseModel):
    id: str
    tool: str
    args: dict
    depends_on: list[str] = []


class Plan(BaseModel):
    steps: list[Step]


async def run_tool(step: Step) -> object:
    await asyncio.sleep(LATENCY)  # 模拟调外部服务的耗时
    return json.loads(_shop.execute(step.tool, json.dumps(step.args, ensure_ascii=False)))


async def execute(plan: Plan) -> dict[str, object]:
    done: dict[str, object] = {}
    pending = list(plan.steps)
    batch_no = 0
    while pending:
        ready = [s for s in pending if all(d in done for d in s.depends_on)]
        if not ready:
            raise ValueError("计划里有循环依赖或依赖了不存在的步骤")
        batch_no += 1
        print(f"  第 {batch_no} 批并发执行：{[s.id + ' ' + s.tool for s in ready]}")
        results = await asyncio.gather(*(run_tool(s) for s in ready))
        done.update({s.id: r for s, r in zip(ready, results, strict=True)})
        pending = [s for s in pending if s.id not in done]
    return done


async def main() -> None:
    llm = LLM()
    print(llm.describe())
    question = "耳机和保温杯分别多少钱？顺便看看我那单耳机到哪了"

    print("\n==== 1. 规划：模型一次给出全部步骤 ====")
    reply = llm.chat(
        [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": question}],
        response_format={"type": "json_object"},  # JSON 模式，见资料库 04 篇
    )
    try:
        plan = Plan.model_validate_json(reply.content or "")  # 计划也要校验，模型可能给出不合法的 JSON
    except ValidationError as exc:
        print(f"  计划不合法：{exc.errors()[0]['msg']}。真实项目里把这个错误发回给模型，让它重写一次计划")
        return
    for s in plan.steps:
        print(f"  {s.id}: {s.tool}({s.args}) 依赖 {s.depends_on or '无'}")

    print("\n==== 2. 执行：程序按依赖分批并发，不再问模型 ====")
    t0 = time.perf_counter()
    results = await execute(plan)
    elapsed = time.perf_counter() - t0
    print(f"  并发执行耗时 {elapsed:.2f} 秒；如果一个个串行跑，至少要 {LATENCY * len(plan.steps):.1f} 秒")

    print("\n==== 3. 汇总：再调一次模型写回答 ====")
    reply = llm.chat([
        {"role": "system", "content": _shop.SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n{json.dumps(results, ensure_ascii=False)}"},
    ])
    print(f"  {reply.content}")

    print("\n==== 4. 对比 ====")
    print(f"  这里调了 {llm.calls} 次模型；ReAct 做同样的事要 1 + {len(plan.steps)} 次（每个工具一次决策，再加回答）")
    print("  代价：计划定下后不能根据中间结果调整；结果出乎意料时要回到规划重新来")


asyncio.run(main())

# 试试改：
# - 给 s3 加上 "depends_on": ["s1"]，看执行分成了几批、耗时怎么变
# - 让计划依赖一个不存在的步骤，看校验怎么报错
# - 写代码：给计划校验增加依赖环检测，让 s1 依赖 s2、s2 依赖 s1 时执行前报错。
