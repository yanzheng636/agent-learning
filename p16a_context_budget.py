"""资料库 16《Context Engineering》：给上下文算预算，看哪块最占地方，再用三招压下来；顺便看前缀缓存为什么会失效。

uv run p16a_context_budget.py      （不需要模型）

看什么：
1. 一次 Agent 会话进行到第 8 轮时，上下文里各部分各占多少 token
2. 三个手段各省多少：工具返回只留需要的字段、旧的工具结果折叠成一句、历史只留最近几轮
3. 缓存友好的排布：顾客名放在系统提示开头和放在末尾，两位顾客能共用的前缀差多少
   （参考项目的系统提示第一行就是顾客名，资料库 29 篇第 5 节提到了这个问题）
"""

import json

import tiktoken

import _shop

enc = tiktoken.get_encoding("o200k_base")


def n(obj) -> int:
    return len(enc.encode(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)))


def bar(parts: dict) -> int:
    total = sum(n(v) for v in parts.values())
    for name, v in parts.items():
        k = n(v)
        print(f"  {name:<14} {k:>5} token  {'█' * (k * 40 // max(total, 1))}")
    print(f"  {'合计':<14} {total:>5} token")
    return total


# 造一段进行到第 8 轮的会话：每轮一问、一次工具调用、一次回答
RAW_ORDER = {"order_id": "ORD-20261001", **_shop.ORDERS["ORD-20261001"], "internal_cost": 88.5,
             "audit_log": [{"at": "2026-10-18 10:00", "by": "system", "action": "created"}] * 5}
history, tool_msgs = [], []
for i in range(7):
    history += [{"role": "user", "content": f"第 {i + 1} 轮：那笔耳机订单现在什么情况？"}]
    tool_msgs.append({"role": "tool", "content": json.dumps(RAW_ORDER, ensure_ascii=False)})
    history += [tool_msgs[-1], {"role": "assistant", "content": "您的耳机订单已发货，顺丰派送中。"}]

print("==== 1. 优化前：第 8 轮请求的上下文构成 ====")
before = bar({
    "系统提示": _shop.SYSTEM_PROMPT,
    "工具定义": _shop.schemas(),
    "历史对话": [m for m in history if m["role"] != "tool"],
    "历史工具结果": tool_msgs,
    "本轮问题": "那电饭煲呢？",
})

print("\n==== 2. 三个手段 ====")
slim = json.loads(_shop.execute("query_order", '{"order_id": "ORD-20261001"}'))
print(f"  ① 工具返回瘦身：一次结果从 {n(RAW_ORDER)} 降到 {n(slim)} token（只留商品、金额、状态、日期）")
folded = [{"role": "tool", "content": "（第 1 至 6 轮的订单查询结果已折叠：耳机订单已发货，顺丰派送中）"}] + \
         [{"role": "tool", "content": json.dumps(slim, ensure_ascii=False)}]
print(f"  ② 旧结果折叠：7 次工具结果从 {n(tool_msgs)} 降到 {n(folded)} token（只保留最近一次全文）")
recent = [m for m in history if m["role"] != "tool"][-6:]
print(f"  ③ 历史只留最近 3 轮：{n([m for m in history if m['role'] != 'tool'])} 降到 {n(recent)} token")
print("\n  优化后：")
after = bar({"系统提示": _shop.SYSTEM_PROMPT, "工具定义": _shop.schemas(), "历史对话": recent,
             "历史工具结果": folded, "本轮问题": "那电饭煲呢？"})
print(f"  从 {before} 降到 {after} token，少了 {100 - after * 100 // before}%。")
print("  注意折叠和截断都可能丢信息，要保住目标、约束和已确认的事实")

print("\n==== 3. 缓存友好的排布：两位顾客能共用多长的前缀 ====")


def shared_prefix(a: str, b: str) -> int:
    ta, tb = enc.encode(a), enc.encode(b)
    k = 0
    while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
        k += 1
    return k


tools_text = json.dumps(_shop.schemas(), ensure_ascii=False)
body = _shop.SYSTEM_PROMPT.split("\n", 1)[1]
name_first = lambda name: f"你是「小店」的智能客服，正在为顾客{name}服务。\n{body}\n{tools_text}"  # noqa: E731
name_last = lambda name: f"你是「小店」的智能客服。\n{body}\n{tools_text}\n当前顾客：{name}"  # noqa: E731
for label, build in [("顾客名放在开头", name_first), ("顾客名放在末尾", name_last)]:
    a, b = build("张三"), build("李四")
    print(f"  {label}：总长 {n(a)} token，两位顾客共同的前缀 {shared_prefix(a, b)} token")
print("  Prompt 缓存按前缀匹配：变化的内容放得越靠前，能共用的部分越少")
print("  （这里把工具定义接在系统提示后面示意；各家把工具定义放在前缀里的位置不同，")
print("    缓存还有最低长度要求，见资料库 05 篇）")

# 试试改：
# - 把历史保留的轮数从 3 改成 1，看还能不能回答“那电饭煲呢”这种依赖上文的问题（提示：要靠摘要保住上文）
# - 往 SYSTEM_PROMPT 开头加一行当前时间，看第 3 段共用的前缀变成多少
# - 写代码：实现按完整对话轮裁剪的函数，保证 assistant.tool_calls 与 tool 结果不会被拆散。
