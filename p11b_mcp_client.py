"""资料库 11《MCP》：宿主程序启动 p11a 这个 MCP Server，发现它的工具，交给模型用。

uv run p11b_mcp_client.py

看什么：
1. 宿主用子进程启动 Server（stdio 传输），并通过环境变量告诉它当前顾客是谁
2. 发现：列出工具和资源，看 Server 自动生成的输入、输出 Schema
3. 转换：把 MCP 的工具定义转成模型接口要的格式，这一步是宿主的活，模型不知道 MCP 的存在
4. 循环：模型要调用工具时，宿主转发给 Server，再把结果回填给模型
5. 出错：Server 抛出的业务错误以 is_error 的结果返回，宿主照样回填
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from _llm import LLM

SERVER = Path(__file__).with_name("p11a_mcp_server.py")


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env={"SHOP_USER": "u1"})
    async with Client(params) as client:
        print(f"==== 1. 连上了 Server：{client.server_info.name}，协议版本 {client.protocol_version} ====")

        print("\n==== 2. 发现：有哪些工具和资源 ====")
        mcp_tools = (await client.list_tools()).tools
        for t in mcp_tools:
            print(f"  工具 {t.name}：{t.description}\n    输入 {json.dumps(t.input_schema, ensure_ascii=False)}")
            print(f"    输出 {json.dumps(t.output_schema, ensure_ascii=False)[:120]}…")
        for r in (await client.list_resources()).resources:
            content = (await client.read_resource(r.uri)).contents[0]
            print(f"  资源 {r.uri}：{getattr(content, 'text', '')}")

        print("\n==== 3. 转换：MCP 工具定义 → 模型接口的工具格式 ====")
        tools = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in mcp_tools
        ]
        print(f"  转好了 {len(tools)} 个：{[t['function']['name'] for t in tools]}")

        print("\n==== 4. 循环：模型调工具，宿主转发给 Server ====")
        llm = LLM()
        print(f"  {llm.describe()}")
        messages = [{"role": "user", "content": "我最近那单到哪了？"}]
        for _ in range(5):
            reply = llm.chat(messages, tools)
            messages.append(reply.as_message())
            if not reply.tool_calls:
                print(f"  回答：{reply.content}")
                break
            for c in reply.tool_calls:
                result = await client.call_tool(c.name, c.args())
                content = json.dumps(result.structured_content, ensure_ascii=False) if result.structured_content \
                    else result.content[0].text  # type: ignore[union-attr]
                messages.append({"role": "tool", "tool_call_id": c.id, "content": content})
                print(f"  {c.name}({c.arguments}) → {content}")

        print("\n==== 5. 出错：查一个不属于当前顾客的订单 ====")
        result = await client.call_tool("track_shipment", {"order_id": "ORD-20261003"})
        print(f"  is_error={result.is_error}，内容：{result.content[0].text}")  # type: ignore[union-attr]
        print("  宿主把这段错误原样回填给模型，模型据此请顾客核对订单号")


asyncio.run(main())

# 试试改：
# - 把 env 里的 SHOP_USER 改成 u2，再看第 4 段查到的是谁的订单
# - 在 p11a 里加一个 search_knowledge 工具，不改这个文件，看第 2 段能不能自动发现它
# - 写代码：实现工具名称白名单，只向模型暴露查询工具，打印发现数量与实际暴露数量。
