# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.11", "tiktoken>=0.9"]
# ///
"""资料库 05《调用约束：Token、成本、延迟与流式》：一次请求的 token 从哪来，Agent 为什么越跑越贵。

uv run p05a_count_tokens.py      （不需要模型；第一次运行 tiktoken 会下载一个编码文件）

看什么：
1. 同样意思的中文和英文各占多少 token
2. 一次请求里系统提示、工具定义、历史、本轮输入各占多少
3. Agent 每走一步都要把前面所有内容重新发一遍，输入 token 怎么累加
4. 按单价算一次对话花多少钱

token 数用 OpenAI 的 o200k_base 编码统计；DeepSeek、通义千问等的分词器不同，数字会有出入，只看量级和比例。
"""

import json

import tiktoken

import _shop

enc = tiktoken.get_encoding("o200k_base")


def n(obj) -> int:
    return len(enc.encode(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)))


# ---- 1. 中英文对比 ----
print("==== 1. 同样意思，中英文各几个 token ====")
for zh, en in [
    ("我的订单到哪了？", "Where is my order?"),
    ("七天无理由退货的运费由谁承担？", "Who pays the shipping for a 7-day no-reason return?"),
    ("饕餮", "gluttonous"),
]:
    print(f"  {zh:<20} {n(zh):>3} 个   |  {en:<52} {n(en):>3} 个")
print("  生僻字可能被拆成好几个 token；用户的中文问法要在自己的模型上实测")

# ---- 2. 一次请求的构成 ----
print("\n==== 2. 小店客服第 6 轮对话时，一次请求的输入构成 ====")
history = []
for i in range(5):  # 前 5 轮，每轮一问一答
    history += [
        {"role": "user", "content": f"第 {i + 1} 个问题：我那单耳机什么时候能到？能顺便看下保修吗？"},
        {"role": "assistant",
         "content": "您的耳机订单已由顺丰派送中，预计今天送达；耳机保修半年，人为损坏不在保修范围内。"},
    ]
parts = {
    "系统提示": _shop.SYSTEM_PROMPT,
    "工具定义（5 个）": _shop.schemas(),
    "历史（5 轮）": history,
    "本轮问题": "那电饭煲呢？",
}
total = sum(n(v) for v in parts.values())
for name, v in parts.items():
    k = n(v)
    print(f"  {name:<12} {k:>5} token  {'█' * (k * 40 // total)} {k * 100 // total}%")
print(f"  合计 {total} token。工具定义每次请求都要带，工具越多这块越大（见资料库 09、12 篇的按需加载）")

# ---- 3. Agent 的多步累加 ----
print("\n==== 3. 一轮回答里 Agent 走了 4 步，每步都要重发前面的全部内容 ====")
context = total
tool_result = json.dumps(_shop.ORDERS["ORD-20261001"], ensure_ascii=False) * 3  # 一个没瘦身的工具返回，约几百字
cumulative = 0
for step in range(1, 5):
    cumulative += context
    print(f"  第 {step} 步：本次输入 {context:>5} token，累计输入 {cumulative:>6} token")
    context += n(tool_result) + 30  # 加上这一步的工具调用和结果
print("  输入 token 随步数近似平方增长：缩短工具返回、让前缀命中 Prompt 缓存最划算（见资料库 29 篇）")

# ---- 4. 算钱 ----
print("\n==== 4. 按单价算这一轮的花费 ====")
PRICE_IN, PRICE_OUT = 2.0, 8.0  # 示例单价：每百万 token 多少元，按你用的模型的官网价格改
out_tokens = 4 * 60  # 每步输出约 60 token
cost = cumulative / 1e6 * PRICE_IN + out_tokens / 1e6 * PRICE_OUT
print(f"  输入 {cumulative} token × {PRICE_IN} 元/百万 + 输出 {out_tokens} token × {PRICE_OUT} 元/百万 ≈ {cost:.4f} 元")
print("  Agent 的成本通常由输入主导；输出单价虽高，量小")

# 试试改：
# - 把工具结果从 *3 改成 *1，看累计输入少了多少
# - 把 _shop.schemas() 换成只带两个工具，看工具定义的占比
# - 把单价换成你常用模型的价格，算一天一万轮对话要花多少
# - 写代码：编写 cost(input_tokens, cached_tokens, output_tokens) 函数，分别按缓存与非缓存单价计费。
