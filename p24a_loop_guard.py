"""资料库 24《Agent 为什么翻车》的死循环：三道兜底——步数上限、重复失败检测、token 预算。

uv run p24a_loop_guard.py

看什么：
1. 只有步数上限：模型如果反复调用工具，最多执行到上限
2. 加上重复失败检测：同一工具、同样参数失败两次，由代码终止本轮，不再执行剩余调用
3. 加上 token 预算：累计输入超过预算就停，把已有进展和缺的信息交代清楚

兜底写在循环里（代码），不是写在提示里请模型自觉：模型负责决策，系统负责守门。
真实模型可能直接追问或成功查到物流，不一定触发重复失败检测或预算上限；观察实际调用记录。
"""

import argparse
import json
from dataclasses import dataclass, field

import _shop
from _llm import LLM


@dataclass
class FailureGuard:
    """参数先解析再排序；无效 JSON 按原文计数，不把解析异常带出循环。"""

    limit: int = 2
    failures: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(self, name: str, arguments: str, result: object) -> bool:
        try:
            canonical = json.dumps(json.loads(arguments or "{}"), sort_keys=True, ensure_ascii=False)
        except json.JSONDecodeError:
            canonical = arguments
        key = (name, canonical)
        if isinstance(result, dict) and "error" in result:
            self.failures[key] = self.failures.get(key, 0) + 1
        else:
            self.failures.pop(key, None)  # 成功后清除该调用的失败计数
        return self.failures.get(key, 0) >= self.limit


def check_guards() -> None:
    guard = FailureGuard()
    error = {"error": "超时"}
    assert not guard.record("lookup", '{"a": 1, "b": 2}', error)
    assert guard.record("lookup", '{"b":2,"a":1}', error)
    assert not guard.record("other", '{"a":1,"b":2}', error)
    assert not guard.record("lookup", '{"a":1,"b":3}', error)
    assert not guard.record("lookup", '{"a":1,"b":2}', {"status": "ok"})
    assert not guard.record("lookup", '{"a":1,"b":2}', error)
    assert not guard.record("lookup", "invalid", error)
    assert guard.record("lookup", "invalid", error)
    print("重复失败计数回归检查通过（8 项；不调用模型）")


def guarded_loop(*, max_steps: int, detect_repeat: bool, token_budget: int | None) -> dict:
    llm = LLM()
    messages = [{"role": "system", "content": _shop.SYSTEM_PROMPT}, {"role": "user", "content": "我的快递到哪了"}]
    tools = _shop.schemas()
    guard = FailureGuard()
    outcome = "本轮已到步数上限，请补充订单号后重试，或联系人工客服。"
    stopped_by = "步数上限"
    for step in range(1, max_steps + 1):
        # 调用前检查，所以实际花费可能超出预算一次调用的量。
        if token_budget is not None and llm.prompt_tokens >= token_budget:
            outcome = f"预算用完（已用 {llm.prompt_tokens} token），本轮查询尚未完成，请联系人工客服"
            stopped_by = "token 预算"
            break
        reply = llm.chat(messages, tools)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            outcome = reply.content or ""
            stopped_by = "模型结束（保护未触发）"
            break
        for c in reply.tool_calls:
            result = json.loads(_shop.execute(c.name, c.arguments))
            should_stop = detect_repeat and guard.record(c.name, c.arguments, result)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, ensure_ascii=False)})
            print(f"    第 {step} 步 {c.name}({c.arguments}) → {result['error'][:60] if 'error' in result else '成功'}")
            if should_stop:
                # 直接返回，阻止同一回复中剩余的工具调用；不会把未配对历史继续发给模型。
                return {"模型调用": llm.calls, "累计输入 token": llm.prompt_tokens,
                        "停止原因": "重复失败", "结果": "同一查询已失败两次，本轮已停止。请核对订单号或稍后再试。"}
    return {"模型调用": llm.calls, "累计输入 token": llm.prompt_tokens, "停止原因": stopped_by, "结果": outcome}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-guards", action="store_true", help="只运行本地保护规则回归检查")
    args = parser.parse_args()
    if args.check_guards:
        check_guards()
        return
    summary = {}
    print("==== 1. 只有步数上限（10 步）====")
    summary["只有步数上限"] = guarded_loop(max_steps=10, detect_repeat=False, token_budget=None)
    print("\n==== 2. 加上重复失败检测 ====")
    summary["加重复失败检测"] = guarded_loop(max_steps=10, detect_repeat=True, token_budget=None)
    print("\n==== 3. 只有 token 预算（3000）====")
    summary["只有 token 预算"] = guarded_loop(max_steps=10, detect_repeat=False, token_budget=3000)

    print("\n==== 对比 ====")
    for name, r in summary.items():
        print(f"  {name:<10} 调用 {r['模型调用']:>2} 次，累计输入 {r['累计输入 token']:>6} token\n"
              f"    停止原因：{r['停止原因']}；给顾客：{r['结果']}")
    print("重复失败由代码终止；真实模型若直接完成或追问，保护可能不触发。预算允许超出一次调用的量。")


if __name__ == "__main__":
    main()

# 试试改：
# - 给 p07a 的循环接入 FailureGuard；触发后禁止执行同一回复里的剩余工具，并打印停止原因。
# - 把 FailureGuard.limit 从 2 改成 3，并同步修改回归断言。
# - 实现按“工具名 + 错误类型”计数的另一种策略，比较参数变化时会不会误拦合法查询。
