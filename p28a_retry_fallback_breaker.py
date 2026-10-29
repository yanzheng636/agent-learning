"""资料库 28《模型路由与 LLM 网关》：按错误类型重试、切备用模型、熔断。

uv run p28a_retry_fallback_breaker.py      （不调真模型：两个“模型服务”是模拟的，按剧本返回成功或各种错误）

看什么：
1. 429 限流：按服务端给的等待时间等一下再试
2. 400 参数错误：不重试，重试也一样错
3. 5xx 和超时：有限次数重试，间隔指数增长加随机抖动；还不行就切到备用模型
4. 熔断：主模型连续失败 3 次后，30 秒内不再打扰它，请求直接走备用；30 秒后放一个请求试探，成功就恢复

时间是模拟的（虚拟时钟），不用真的等，几毫秒跑完。
"""

import random
from dataclasses import dataclass, field

rng = random.Random(0)


class Clock:
    now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds


clock = Clock()


class ModelError(Exception):
    def __init__(self, status: int, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status, self.retry_after = status, retry_after


@dataclass
class FakeProvider:
    name: str
    script: list[str]  # 每次调用按顺序取一个结果：ok、429、400、500、timeout；用完以后一直 ok

    def complete(self, prompt: str) -> str:
        outcome = self.script.pop(0) if self.script else "ok"
        clock.sleep(0.8)  # 一次调用耗时
        if outcome == "ok":
            return f"{self.name} 的回答"
        if outcome == "timeout":
            raise ModelError(504)
        raise ModelError(int(outcome), retry_after=2.0 if outcome == "429" else None)


@dataclass
class Breaker:
    threshold: int = 3
    cooldown: float = 30.0
    failures: int = 0
    opened_at: float | None = None
    log: list[str] = field(default_factory=list)

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if clock.now - self.opened_at >= self.cooldown:
            self.log.append(f"t={clock.now:.1f}s 冷却结束，放一个请求试探（半开）")
            return True
        return False

    def record(self, ok: bool) -> None:
        if ok:
            if self.opened_at is not None:
                self.log.append(f"t={clock.now:.1f}s 试探成功，熔断关闭")
            self.failures, self.opened_at = 0, None
            return
        self.failures += 1
        if self.opened_at is not None:  # 半开时试探失败：重新打开，再等一个冷却期
            self.log.append(f"t={clock.now:.1f}s 试探失败，熔断重新打开")
            self.opened_at = clock.now
        elif self.failures >= self.threshold:
            self.log.append(f"t={clock.now:.1f}s 连续失败 {self.failures} 次，熔断打开 {self.cooldown:.0f} 秒")
            self.opened_at = clock.now


def call_with_retry(provider: FakeProvider, prompt: str, breaker: Breaker | None = None, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            result = provider.complete(prompt)
            if breaker:
                breaker.record(True)
            return result
        except ModelError as e:
            if breaker:
                breaker.record(False)
            if e.status == 400:
                print(f"    t={clock.now:.1f}s {provider.name} 返回 400，参数有误，不重试")
                raise
            if attempt == max_retries or (breaker and not breaker.allow()):
                raise
            wait = e.retry_after or (0.5 * 2**attempt + rng.uniform(0, 0.3))  # 指数退避加抖动
            print(f"    t={clock.now:.1f}s {provider.name} 返回 {e.status}，{wait:.1f} 秒后第 {attempt + 1} 次重试")
            clock.sleep(wait)
    raise AssertionError("unreachable")


def ask(primary: FakeProvider, backup: FakeProvider, breaker: Breaker, prompt: str) -> str:
    if breaker.allow():
        try:
            return call_with_retry(primary, prompt, breaker)
        except ModelError as e:
            if e.status == 400:
                return "参数错误，记录下来修代码，不切换模型"
            print(f"    t={clock.now:.1f}s 主模型重试后仍失败，切到备用模型")
    else:
        print(f"    t={clock.now:.1f}s 熔断中，直接走备用模型")
    return call_with_retry(backup, prompt)


print("==== 1. 429 限流：等服务端说的时间再试 ====")
clock.now = 0.0
primary = FakeProvider("主模型", ["429", "ok"])
print(f"  结果：{ask(primary, FakeProvider('备用模型', []), Breaker(), '你好')}")

print("\n==== 2. 400 参数错误：不重试 ====")
clock.now = 0.0
primary = FakeProvider("主模型", ["400"])
print(f"  结果：{ask(primary, FakeProvider('备用模型', []), Breaker(), '你好')}")

print("\n==== 3. 主模型故障：重试、切备用、熔断、恢复 ====")
clock.now = 0.0
primary = FakeProvider("主模型", ["500", "timeout", "500"])  # 连续出错三次，之后恢复
backup = FakeProvider("备用模型", [])
breaker = Breaker()
for i, gap in enumerate([0, 5, 5, 35]):  # 四个请求，最后一个在 35 秒后到来
    clock.sleep(gap)
    print(f"  请求 {i + 1}（t={clock.now:.1f}s）")
    print(f"    → {ask(primary, backup, breaker, '查订单')}")
print("  熔断器的记录：")
for line in breaker.log:
    print(f"    {line}")

# 试试改：
# - 把 max_retries 改成 0，看第 3 段是不是更快切到备用
# - 把冷却时间改成 60 秒，看第 4 个请求走哪个模型
# - 想一想：流式输出已经推了一半给用户时出错，还能不能切备用（资料库 28 篇 2.2 节）
# - 写代码：加入半开状态的单探针限制，并验证探针失败后重新熔断、成功后恢复。
