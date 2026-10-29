"""资料库 08《常用编排模式》里的链式和路由：流程由代码定，模型只负责其中几步。

uv run p08a_chain_router.py

看什么：
1. 链式：抽取字段 → 代码检查（关卡）→ 写回复。关卡不过就停下追问，不让错误流到下一步
2. 路由：先判断问题属于哪一类，再交给对应的处理流程，每类用自己的提示和工具
3. 两种模式里，调用顺序都写在代码里，不是模型决定的：这是 Workflow，不是 Agent（资料库 02 篇）
"""

import json

from pydantic import BaseModel, ValidationError

import _shop
from _llm import LLM


class AfterSale(BaseModel):
    order_id: str | None
    problem: str
    want: str  # 退货 / 换货 / 维修 / 咨询


llm = LLM()
print(llm.describe())


def ask(system: str, user: str) -> str:
    return llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}]).content or ""


# ---- 1. 链式 ----
def after_sale_chain(message: str) -> str:
    raw = ask('抽取售后信息，只输出 JSON：'
              '{"order_id": 订单号或 null, "problem": "...", "want": "退货/换货/维修/咨询"}', message)
    try:
        info = AfterSale.model_validate_json(raw)
    except ValidationError:
        return "（第 1 步输出不合法，重试一次或转人工）"
    print(f"  第 1 步 抽取：{info.model_dump()}")
    # 关卡：用代码检查，不靠模型
    if info.order_id is None:
        return "关卡没过：缺订单号，追问“请提供订单号”，后面的步骤不执行"
    if info.order_id not in _shop.ORDERS or _shop.ORDERS[info.order_id]["user_id"] != _shop.CURRENT_USER:
        return f"关卡没过：{info.order_id} 不是当前顾客的订单，后面的步骤不执行"
    print("  第 2 步 关卡：订单号存在且属于当前顾客")
    prompt = f"写售后回复：顾客想{info.want}，订单 {info.order_id}，语气友好，不承诺政策外的赔付"
    return "第 3 步 写回复：" + ask(prompt, message)


print("\n==== 1. 链式：抽取 → 关卡 → 写回复 ====")
for msg in [
    "订单 ORD-20261001 的耳机左边没声音，我要换货",
    "耳机左边没声音，我要退货",
    "ORD-20261003 的杯子漏水，我要退货",
]:
    print(f"顾客：{msg}")
    print(f"  结果：{after_sale_chain(msg)}")


# ---- 2. 路由 ----
def handle_order(q: str) -> str:
    # 真实项目里这个分支是一个带查订单、查物流工具的 Agent 循环（见 p07a），这里只示意
    return f"交给订单流程：{ask('订单助手：先调用工具查到数据再回答', q)}"


def handle_policy(q: str) -> str:
    hits = json.loads(_shop.execute("search_knowledge", json.dumps({"query": q}, ensure_ascii=False)))
    return f"交给政策流程（先检索 {hits[0]['source']}，按原文回答）：{ask('政策助手：只按给出的资料回答', q)}"


def handle_chitchat(q: str) -> str:
    return "闲聊：固定回复“我是小店客服，可以帮您查订单、物流和售后政策”，不调模型"


ROUTES = {"订单": handle_order, "政策": handle_policy, "闲聊": handle_chitchat}

print("\n==== 2. 路由：先分类，再交给对应流程 ====")
for q in ["我的快递到哪了", "七天无理由退货运费谁出", "你好呀"]:
    label = ask("分类：把问题分成 订单 / 政策 / 闲聊 三类之一，只输出类别名", q).strip()
    handler = ROUTES.get(label, handle_chitchat)  # 分类结果不在预期里时走兜底分支
    print(f"顾客：{q}\n  分类：{label} → {handler(q)}")

print(f"\n共调用模型 {llm.calls} 次，每次调用的位置和顺序都由代码决定")

# 试试改：
# - 加一个“投诉”类别和对应的处理函数，看路由表怎么扩展
# - 把关卡改成“金额超过 300 元的退货转人工”，看链式流程在哪一步停下
# - 写代码：为无法识别的类别实现默认转人工分支，并记录原始分类与路由目标。
