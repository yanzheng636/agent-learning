"""示例共用的小店数据和工具：和参考项目 shop-assistant 的五个只读工具同名，数据放在内存里。

当前顾客由调用方通过 user_id 传入，模型改不了，所以查不到别人的订单（资料库 10、27 篇讲的“身份由系统注入”）。
工具参数用 Pydantic 定义，JSON Schema 由它生成，校验失败时把哪个字段错了告诉模型（p09a 拆开讲了这一套）。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CURRENT_USER = "u1"

ORDERS: dict[str, dict[str, Any]] = {
    "ORD-20261001": {
        "user_id": "u1", "product": "降噪蓝牙耳机", "amount": 199.0, "status": "已发货",
        "created_at": "2026-10-18", "tracking_no": "SF1001",
        "address": "上海市浦东新区张江路 88 号", "phone": "13800001111",
    },
    "ORD-20261002": {
        "user_id": "u1", "product": "3L 电饭煲", "amount": 349.0, "status": "待发货",
        "created_at": "2026-10-22", "tracking_no": None,
        "address": "上海市浦东新区张江路 88 号", "phone": "13800001111",
    },
    "ORD-20261003": {
        "user_id": "u2", "product": "保温杯", "amount": 89.0, "status": "已签收",
        "created_at": "2026-10-10", "tracking_no": "YT2002",
        "address": "杭州市西湖区文三路 1 号", "phone": "13900002222",
    },
}

SHIPMENTS: dict[str, dict[str, Any]] = {
    "SF1001": {"carrier": "顺丰", "events": [
        ["2026-10-19 09:12", "已揽收"], ["2026-10-20 21:40", "到达上海转运中心"], ["2026-10-21 08:05", "派送中"],
    ]},
    "YT2002": {"carrier": "圆通", "events": [["2026-10-11 10:00", "已揽收"], ["2026-10-13 15:20", "已签收"]]},
}

PRODUCTS = [
    {"name": "降噪蓝牙耳机", "price": 199.0, "stock": 35},
    {"name": "入耳式有线耳机", "price": 59.0, "stock": 120},
    {"name": "保温杯", "price": 89.0, "stock": 0},
    {"name": "3L 电饭煲", "price": 349.0, "stock": 12},
]

KNOWLEDGE = [
    {"id": "policy/退换货政策",
     "text": "签收后 7 天内可无理由退货，商品需完好；无理由退货的运费由顾客承担，质量问题的运费由小店承担。"},
    {"id": "policy/配送说明", "text": "下单后 48 小时内发货，默认发顺丰，偏远地区 3 到 5 天送达。"},
    {"id": "policy/保修", "text": "电器类商品保修一年，耳机保修半年，人为损坏不在保修范围内。"},
]

SYSTEM_PROMPT = """\
你是「小店」的智能客服。
- 用简洁、友好的中文回答，一般不超过三句话。
- 只回答和本店商品、订单、物流、售后有关的问题。
- 涉及订单、物流、价格和库存，先调用工具查到数据再回答；不知道的不要编造。
- 订单号的格式是 ORD- 加 8 位数字。顾客没给订单号时，先查看他最近的订单，或者请他提供。"""


class _Args(BaseModel):

    model_config = ConfigDict(extra="forbid")  # 模型多传了字段也算错，免得它以为传了就生效


class NoArgs(_Args):
    pass


class OrderArgs(_Args):
    order_id: str = Field(pattern=r"^ORD-\d{8}$", description="订单号，格式是 ORD- 加 8 位数字，如 ORD-20261001")


class KeywordArgs(_Args):
    keyword: str = Field(min_length=1, description="商品关键词，如“耳机”")


class QueryArgs(_Args):
    query: str = Field(min_length=1, description="要查的问题，如“退货运费谁出”")


def _own_order(user_id: str, order_id: str) -> dict[str, Any] | None:
    order = ORDERS.get(order_id)
    return order if order and order["user_id"] == user_id else None  # 别人的订单和不存在一样处理


def list_my_orders(user_id: str, _: NoArgs) -> Any:
    return [
        {"order_id": k, "product": o["product"], "status": o["status"], "created_at": o["created_at"]}
        for k, o in ORDERS.items()
        if o["user_id"] == user_id
    ]


def query_order(user_id: str, a: OrderArgs) -> Any:
    order = _own_order(user_id, a.order_id)
    if order is None:
        return {"error": f"没有找到订单 {a.order_id}，请顾客核对订单号"}
    return {"order_id": a.order_id, **{k: order[k] for k in ("product", "amount", "status", "created_at")}}


def track_shipment(user_id: str, a: OrderArgs) -> Any:
    order = _own_order(user_id, a.order_id)
    if order is None:
        return {"error": f"没有找到订单 {a.order_id}，请顾客核对订单号"}
    if order["tracking_no"] is None:
        return {"error": f"订单 {a.order_id} 还没有发货，暂时没有物流信息"}
    ship = SHIPMENTS[order["tracking_no"]]
    return {"order_id": a.order_id, "carrier": ship["carrier"], "latest": ship["events"][-1]}


def search_products(user_id: str, a: KeywordArgs) -> Any:
    return [p for p in PRODUCTS if a.keyword in p["name"]] or {"error": f"没有找到和“{a.keyword}”有关的商品"}


def search_knowledge(user_id: str, a: QueryArgs) -> Any:
    def score(doc: dict[str, str]) -> int:  # 按共同的字数粗排，真实项目用检索，见资料库 18 篇
        return len(set(a.query) & set(doc["text"]))

    hits = sorted(KNOWLEDGE, key=score, reverse=True)[:2]
    return [{"source": d["id"], "text": d["text"]} for d in hits]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[str, Any], Any]


TOOLS = {
    t.name: t
    for t in [
        Tool("list_my_orders", "列出当前顾客最近的订单（订单号、商品、状态、下单日期）。顾客没给订单号时先用它。",
             NoArgs, list_my_orders),
        Tool("query_order", "查一笔订单的商品、金额、状态。只能查当前顾客自己的订单。", OrderArgs, query_order),
        Tool("track_shipment", "查一笔已发货订单的快递公司和最新物流。", OrderArgs, track_shipment),
        Tool("search_products", "按关键词搜索店里的商品，返回价格和库存。", KeywordArgs, search_products),
        Tool("search_knowledge", "查店铺资料：退换货、保修、配送等规定。回答政策类问题前先查。",
             QueryArgs, search_knowledge),
    ]
}


def schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """把工具转成 Chat Completions 接口要的格式；参数的 JSON Schema 由 Pydantic 生成。"""
    return [
        {"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.args_model.model_json_schema(),
        }}
        for t in TOOLS.values()
        if names is None or t.name in names
    ]


def args_error(exc: ValidationError) -> str:
    """把参数校验错误写成模型能照着改的工具结果：哪个字段、错在哪。示例里自定义的工具也用它。"""
    problems = "；".join(f"{'.'.join(map(str, e['loc'])) or '参数'}：{e['msg']}" for e in exc.errors())
    return json.dumps({"error": f"参数不合法，请修改后重试。{problems}"}, ensure_ascii=False)


def execute(name: str, arguments: str, user_id: str = CURRENT_USER) -> str:
    """执行一次工具调用。出了问题不抛异常，而是把原因写进结果交给模型，让它能改正或换个做法。"""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"没有名为 {name} 的工具"}, ensure_ascii=False)
    try:
        args = tool.args_model.model_validate_json(arguments or "{}")
    except ValidationError as exc:
        return args_error(exc)
    return json.dumps(tool.func(user_id, args), ensure_ascii=False)
