# /// script
# requires-python = ">=3.12"
# dependencies = ["langchain-openai>=1.7,<2", "python-dotenv>=1.1,<2", "pydantic-settings>=2.10,<3"]
# ///
"""资料库 13 第 2 节的结构化输出配套扩展：自然语言 → Pydantic → 业务关卡。

uv run p13c_langchain_structured.py

看什么：Schema 校验与业务权限校验是两件事；字段合法不代表允许操作订单。
模型必须支持 tool calling；这里显式使用 function_calling，不要求原生 JSON Schema 模式。
"""

import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import _shop
from _langchain import make_model


class AfterSale(BaseModel):
    order_id: str | None = Field(description="用户明确提供的订单号，未提供则为 null", pattern=r"^ORD-\d{8}$")
    problem: str = Field(description="顾客描述的问题")
    want: Literal["退货", "换货", "维修", "咨询"]


def main() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "抽取售后信息，不要推测订单号。未明确表达售后诉求时，want 填咨询。"),
        ("human", "{message}"),
    ])
    extractor = prompt | make_model().with_structured_output(
        AfterSale, method="function_calling", include_raw=True,
    )

    print("==== 抽取 → Schema 校验 → 订单归属校验 ====")
    for message in [
        "ORD-20261001 的耳机左边没声音，我要换货",
        "耳机坏了，我想退货",
        "ORD-20261003 的杯子漏水，我要退货",
    ]:
        print(f"\n顾客：{message}")
        result = extractor.invoke({"message": message})
        info = result["parsed"]
        if result["parsing_error"] is not None or info is None:
            print("  模型没有返回合法结构，应重试或转人工；本例不继续操作")
            continue
        print(f"  抽取结果（{type(info).__name__}）：{info.model_dump()}")
        if info.order_id is None:
            print("  业务关卡：缺少订单号，请顾客补充")
            continue
        order = json.loads(_shop.execute("query_order", json.dumps({"order_id": info.order_id})))
        if "error" in order:
            print(f"  业务关卡：{order['error']}")
            continue
        print(f"  业务关卡通过：{order['product']}，诉求是{info.want}；仅登记，不执行退款")


if __name__ == "__main__":
    main()

# 试试改：加一个 severity: Literal["普通", "紧急"] 字段，观察 Schema 如何变。
# 模型调用失败（例如鉴权、超时）会直接报错；include_raw 只帮助检查解析结果。
# - 写代码：增加 severity 枚举字段，并用业务函数把紧急诉求路由到人工处理。
