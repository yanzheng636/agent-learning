"""LangChain 系列共用的模型配置和工具适配；业务数据仍来自 _shop。"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import _shop
from setting import settings


def make_model() -> ChatOpenAI:
    """读取统一 settings 创建模型，配置加载由 setting.py 负责。"""
    if not settings.llm_api_key or not settings.llm_model:
        raise RuntimeError("请在 .env 或环境变量中设置 LLM_API_KEY 和 LLM_MODEL；第三方服务还需设置 LLM_BASE_URL")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or "https://api.openai.com/v1",
        timeout=60,
        max_retries=1,
        use_responses_api=False,  # 使用兼容 OpenAI 的 Chat Completions，避免供应商专属搜索工具。
    )


@tool
def list_my_orders() -> str:
    """列出当前顾客的订单。没给订单号时先调用它，不要猜订单号。"""
    return _shop.execute("list_my_orders", "{}")


@tool(args_schema=_shop.OrderArgs)
def query_order(order_id: str) -> str:
    """查询当前顾客一笔订单的商品、金额和状态。"""
    return _shop.execute("query_order", _shop.OrderArgs(order_id=order_id).model_dump_json())


@tool(args_schema=_shop.OrderArgs)
def track_shipment(order_id: str) -> str:
    """查询当前顾客一笔订单的快递公司和最新物流。"""
    return _shop.execute("track_shipment", _shop.OrderArgs(order_id=order_id).model_dump_json())


# 身份由 _shop.execute 在程序侧注入；工具 Schema 中没有 user_id。
SHOP_TOOLS = [list_my_orders, query_order, track_shipment]
