# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp>=2.3", "pydantic>=2.11"]
# ///
"""资料库 11《MCP》：把小店的查询能力做成一个 MCP Server。

不要直接运行它：它通过标准输入输出（stdio）和客户端说话，单独跑起来只会静静等输入。用 p11b 启动并调用它：

    uv run p11b_mcp_client.py

也可以用官方的调试界面看：uv run --with mcp mcp dev p11a_mcp_server.py

看什么：
1. 工具就是带类型注解的函数：参数类型就是 JSON Schema，docstring 就是描述
2. 返回 TypedDict，客户端拿到的是结构化结果（structured_content），并能看到输出的 Schema
3. 出错抛 ToolError，错误信息原样交给模型；别的异常只会给一句笼统的错误
4. 当前顾客由启动它的宿主程序通过环境变量 SHOP_USER 告诉它，模型传不了
5. 资源（resource）：只读的资料，由宿主决定要不要放进上下文

用的是 Python SDK 2.x（截至 2026 年 10 月为 2.3）：类名是 MCPServer，1.x 里叫 FastMCP。
"""

import json
import os
from typing import TypedDict

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import _shop

USER = os.environ.get("SHOP_USER", "")  # 宿主启动时注入；没有就谁的订单都查不到

server = MCPServer("小店")


class OrderBrief(TypedDict):
    order_id: str
    product: str
    status: str
    created_at: str


class Shipment(TypedDict):
    order_id: str
    carrier: str
    latest: list[str]


def _call(name: str, **args: str) -> object:
    result = json.loads(_shop.execute(name, json.dumps(args, ensure_ascii=False), user_id=USER))
    if isinstance(result, dict) and "error" in result:
        raise ToolError(result["error"])  # 业务错误交给模型，它能据此追问或改参数
    return result


@server.tool()
def list_my_orders() -> list[OrderBrief]:
    """列出当前顾客最近的订单。顾客没给订单号时先用它。"""
    return _call("list_my_orders")  # type: ignore[return-value]


@server.tool()
def track_shipment(order_id: str) -> Shipment:
    """查一笔已发货订单的快递公司和最新物流。order_id 的格式是 ORD- 加 8 位数字。"""
    return _call("track_shipment", order_id=order_id)  # type: ignore[return-value]


@server.resource("policy://returns")
def returns_policy() -> str:
    """小店的退换货政策全文。"""
    return _shop.KNOWLEDGE[0]["text"]


if __name__ == "__main__":
    server.run()  # 默认走 stdio；部署成远程服务用 server.run("streamable-http")

# 试试改：
# - 写代码：用 @server.tool() 实现 search_knowledge，返回带 source 的资料，并用 p11b 验证自动发现。
