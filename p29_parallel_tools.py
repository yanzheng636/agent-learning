# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.11"]
# ///
"""资料库 29《性能与成本优化》：模型一轮给出几个互不依赖的工具调用时，并发执行，耗时从“相加”变成“取最慢”。

uv run p29_parallel_tools.py      （不需要模型；工具加了模拟的网络延迟）

看什么：
1. 串行和并发跑同样三个调用，各花多久
2. 单个工具设超时：慢的那个超时报错，不拖垮其他工具，结果照样回填给模型
3. 有依赖的调用不能并发：先列订单才知道查哪个物流，这种只能分两轮

参考项目 shop-assistant 的 ToolBox.execute_all 就是这么做的：TaskGroup 并发、每个工具单独超时、失败写进结果而不抛异常。
"""

import asyncio
import json
import time

import _shop

LATENCY = {"list_my_orders": 0.4, "search_products": 0.4, "track_shipment": 0.6, "search_knowledge": 0.5}
TOOL_TIMEOUT = 1.0

CALLS = [
    ("search_products", {"keyword": "耳机"}),
    ("track_shipment", {"order_id": "ORD-20261001"}),
    ("search_knowledge", {"query": "耳机保修多久"}),
]


async def run_tool(name: str, args: dict, latency: float | None = None) -> str:
    try:
        async with asyncio.timeout(TOOL_TIMEOUT):
            await asyncio.sleep(LATENCY[name] if latency is None else latency)  # 模拟调外部服务
            return _shop.execute(name, json.dumps(args, ensure_ascii=False))
    except TimeoutError:
        return json.dumps({"error": f"{name} 超时，请稍后再试"}, ensure_ascii=False)  # 失败写进结果，不抛异常


async def serial() -> list[str]:
    return [await run_tool(n, a) for n, a in CALLS]


async def concurrent(latencies: list[float | None] | None = None) -> list[str]:
    latencies = latencies or [None] * len(CALLS)
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_tool(n, a, lat)) for (n, a), lat in zip(CALLS, latencies, strict=True)]
    return [t.result() for t in tasks]


async def main() -> None:
    print("==== 1. 串行 vs 并发 ====")
    for label, fn in [("串行", serial), ("并发", concurrent)]:
        t0 = time.perf_counter()
        results = await fn()
        print(f"  {label}：{time.perf_counter() - t0:.2f} 秒，拿到 {len(results)} 个结果")
    print(f"  各工具耗时 {[LATENCY[n] for n, _ in CALLS]}：串行是相加，并发是取最慢的那个")

    print(f"\n==== 2. 其中一个变慢（2 秒），单个工具超时 {TOOL_TIMEOUT} 秒 ====")
    t0 = time.perf_counter()
    results = await concurrent([None, 2.0, None])
    print(f"  总耗时 {time.perf_counter() - t0:.2f} 秒")
    for (name, _), r in zip(CALLS, results, strict=True):
        print(f"  {name:<16} → {r[:70]}")
    print("  超时的那个以错误回填给模型，另外两个照常可用；模型可以先回答能回答的部分")

    print("\n==== 3. 有依赖的调用只能分轮 ====")
    t0 = time.perf_counter()
    orders = json.loads(await run_tool("list_my_orders", {}))  # 第一轮
    target = next(o["order_id"] for o in orders if o["status"] == "已发货")
    print(f"  第一轮列出 {len(orders)} 笔订单，找到已发货的 {target}，第二轮才知道查哪笔的物流")
    await run_tool("track_shipment", {"order_id": target})  # 第二轮
    print(f"  两轮共 {time.perf_counter() - t0:.2f} 秒；真实的 Agent 里两轮之间还隔着一次模型调用")


asyncio.run(main())

# 试试改：
# - 把 TOOL_TIMEOUT 调到 3 秒，看第 2 段总耗时变成多少
# - 想一想：如果并发的几个调用里有写操作（比如提交退款），超时以后能不能直接重试（资料库 24 篇：幂等）
# - 写代码：实现最大并发数限制；记录同时运行的任务数，验证不会超过配置值。
