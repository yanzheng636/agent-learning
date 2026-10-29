"""资料库 09《工具设计》：用 Pydantic 定义工具参数，生成 JSON Schema、校验模型给的参数、把错误写成模型能改的样子。

uv run p09a_tool_schema_pydantic.py      （不需要模型）

看什么：
1. 一个类同时给出三样东西：发给模型的参数说明、代码里的类型、运行时的校验
2. 模型给的参数五花八门：格式错、多字段、缺字段、JSON 不合法，每种错误都要回填一句模型看得懂的话
3. 返回值要瘦身：同一笔订单，原样返回和只挑需要的字段，token 差多少
4. 好的描述和差的描述放在一起比
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import _shop
from _llm import count_tokens


class CreateRefundDraftArgs(BaseModel):
    """生成退款申请草稿（不会真的退款，顾客确认后才提交）。"""

    model_config = ConfigDict(extra="forbid")  # 多传的字段直接报错，免得模型以为传了就生效

    order_id: str = Field(pattern=r"^ORD-\d{8}$", description="订单号，格式是 ORD- 加 8 位数字，如 ORD-20261001")
    reason: Literal["质量问题", "不想要了", "发错货", "其他"] = Field(description="退款原因，只能是这四个之一")
    amount: float = Field(gt=0, description="申请退款的金额，单位元，不能超过订单金额")


# ---- 1. 一个类，三样东西 ----
print("==== 1. 发给模型的工具定义 ====")
schema = {
    "type": "function",
    "function": {
        "name": "create_refund_draft",
        "description": CreateRefundDraftArgs.__doc__,
        "parameters": CreateRefundDraftArgs.model_json_schema(),
    },
}
print(json.dumps(schema, ensure_ascii=False, indent=2))
print("要点：pattern、枚举、取值范围都写进了 Schema；不允许额外字段（additionalProperties: false）")


# ---- 2. 校验模型给的参数 ----
def validate(arguments: str) -> str:
    """校验通过返回 ok；不通过返回一句给模型看的话：哪个字段、错在哪、该怎么改。"""
    try:
        args = CreateRefundDraftArgs.model_validate_json(arguments)
    except ValidationError as exc:
        problems = "；".join(f"{'.'.join(map(str, e['loc'])) or '参数'}：{e['msg']}" for e in exc.errors())
        return f"参数不合法，请修改后重试。{problems}"
    order = _shop.ORDERS.get(args.order_id)
    if order and args.amount > order["amount"]:  # Schema 表达不了的业务规则，在代码里再查一遍
        return f"退款金额 {args.amount} 超过了订单金额 {order['amount']}，请修改金额"
    return f"ok：{args.model_dump()}"


print("\n==== 2. 模型可能给出的各种参数 ====")
cases = {
    "正确": '{"order_id": "ORD-20261001", "reason": "质量问题", "amount": 199}',
    "订单号格式错": '{"order_id": "20261001", "reason": "质量问题", "amount": 199}',
    "原因不在枚举里": '{"order_id": "ORD-20261001", "reason": "左耳没声音", "amount": 199}',
    "多传了字段": '{"order_id": "ORD-20261001", "reason": "其他", "amount": 50, "urgent": true}',
    "缺字段": '{"order_id": "ORD-20261001"}',
    "超过订单金额": '{"order_id": "ORD-20261001", "reason": "其他", "amount": 999}',
    "JSON 不合法": '{"order_id": "ORD-20261001", reason: 质量问题}',
}
for name, arguments in cases.items():
    print(f"  {name:<8} → {validate(arguments)[:110]}")
print("校验失败时不抛异常，把这句话作为工具结果回填给模型，它通常下一步就能改对")

# ---- 3. 返回值瘦身 ----
print("\n==== 3. 同一笔订单，返回什么 ====")
full = {"order_id": "ORD-20261001", **_shop.ORDERS["ORD-20261001"], "internal_cost": 88.5, "warehouse": "WH-SH-03",
        "audit_log": [{"at": "2026-10-18 10:00", "by": "system", "action": "created"}] * 6}
slim = json.loads(_shop.execute("query_order", '{"order_id": "ORD-20261001"}'))
print(f"  原样返回：{count_tokens(full)} token，含收货地址、电话、内部成本、审计日志")
print(f"  只挑需要的字段：{count_tokens(slim)} token → {slim}")
print("  多出来的字段每一步都要重复计费，还可能把隐私和内部数据带进对话")

# ---- 4. 描述写得好不好 ----
print("\n==== 4. 工具描述：差的和好的 ====")
print("  差：name=order, description=订单接口")
print("  好：name=query_order, description=查一笔订单的商品、金额、状态。只能查当前顾客自己的订单；"
      "顾客没给订单号时先用 list_my_orders")
print("  好的描述说清：做什么、返回什么、什么时候用、和相近的工具怎么区分")

# 试试改：
# - 给 reason 加一个枚举值，看 Schema 怎么变
# - 把 extra="forbid" 去掉，再跑“多传了字段”那一条，看会发生什么
# - 写代码：给工具增加 quantity: int 字段并约束 1～10，为 0、11 和字符串分别添加校验案例。
