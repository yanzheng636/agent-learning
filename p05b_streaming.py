# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "langchain-openai>=1.7,<2",
#   "python-dotenv>=1.1,<2",
#   "pydantic-settings>=2.10,<3",
# ]
# ///
"""资料库 05：LangChain 流式输出、首文本延迟、交错 tool_calls 分片。

uv run p05b_streaming.py
uv run p05b_streaming.py --without-usage   供应商不支持 stream_options 时使用
uv run p05b_streaming.py --check          仅检查分片、截断和非法参数

看什么：ChatOpenAI.stream 返回 AIMessageChunk；用 + 按 index 合并工具分片。
LangChain 的部分 JSON 解析可用于展示，不能直接用于执行：整批结束后必须严格 json.loads。
首事件、首文本、首工具分片分别计时；本例测的是客户端收到事件的时间。
"""

import argparse
import json
from time import perf_counter

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

import _shop
from _langchain import SHOP_TOOLS, make_model


def complete_message(assembled: AIMessageChunk) -> AIMessage:
    parts = assembled.tool_call_chunks
    expected = "tool_calls" if parts else "stop"
    reason = assembled.response_metadata.get("finish_reason")
    if reason != expected:
        raise ValueError(f"流未正常完成（finish_reason={reason!r}），禁止执行工具")
    calls = []
    for part in parts:
        if not part.get("id") or not part.get("name"):
            raise ValueError("缺少完整的工具 id 或名称")
        # 不使用 chunk.tool_calls 中可能由 parse_partial_json 修补出来的参数。
        args = json.loads(part["args"])
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是完整 JSON 对象")
        calls.append({"id": part["id"], "name": part["name"], "args": args, "type": "tool_call"})
    if len({c["id"] for c in calls}) != len(calls):
        raise ValueError("tool_call_id 重复")
    return AIMessage(content=assembled.content, tool_calls=calls, usage_metadata=assembled.usage_metadata)


def stream_reply(model, messages: list, *, include_usage: bool) -> AIMessage:
    assembled = None
    start = perf_counter()
    first_event = first_text = first_tool = None
    for chunk in model.stream(messages, stream_usage=include_usage):
        elapsed = perf_counter() - start
        if first_event is None:
            first_event = elapsed
        assembled = chunk if assembled is None else assembled + chunk
        if chunk.tool_call_chunks and first_tool is None:
            first_tool = elapsed
        if isinstance(chunk.content, str) and chunk.content:
            if first_text is None:
                first_text = elapsed
            print(chunk.content, end="", flush=True)
    if assembled is None:
        raise ValueError("供应商返回空流")
    reply = complete_message(assembled)  # 所有工具都验证完，调用方才会执行第一项
    print()
    for label, value in (("首事件", first_event), ("首文本", first_text), ("首工具分片", first_tool)):
        print(f"  {label}：{value:.3f}s" if value is not None else f"  {label}：本轮没有")
    usage = assembled.usage_metadata
    print(f"  总耗时：{perf_counter() - start:.3f}s；用量：{usage if usage is not None else '供应商未提供'}")
    return reply


def check() -> None:
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"index": 0, "id": "a", "name": "query_", "args": '{"order_'},
            {"index": 1, "id": "b", "name": "track_shipment", "args": '{"order_id":"ORD-'},
        ],
    )
    second = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"index": 1, "id": None, "name": None, "args": '20261001"}'},
            {"index": 0, "id": None, "name": "order", "args": 'id":"ORD-20261002"}'},
        ],
        response_metadata={"finish_reason": "tool_calls"},
    )
    usage = AIMessageChunk(content="", usage_metadata={"input_tokens": 42, "output_tokens": 12, "total_tokens": 54})
    reply = complete_message(first + second + usage)
    assert [(c["id"], c["name"], c["args"]["order_id"]) for c in reply.tool_calls] == [
        ("a", "query_order", "ORD-20261002"),
        ("b", "track_shipment", "ORD-20261001"),
    ]
    assert reply.usage_metadata["input_tokens"] == 42
    for reason, arguments in ((None, "{}"), ("length", "{}"), ("tool_calls", '{"x":'), ("tool_calls", "[]")):
        invalid = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "a", "name": "query_order", "args": arguments},
            ],
            response_metadata={"finish_reason": reason},
        )
        try:
            complete_message(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError((reason, arguments))
    plain = AIMessageChunk(content="你") + AIMessageChunk(content="好", response_metadata={"finish_reason": "stop"})
    assert complete_message(plain).content == "你好" and plain.usage_metadata is None
    print("LangChain 交错分片、工具名分片、用量、截断、非法 JSON、参数类型与文本检查通过；未调用模型。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="请查 ORD-20261001 和 ORD-20261002 的订单及物流。")
    parser.add_argument("--without-usage", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    model = make_model().bind_tools(SHOP_TOOLS)
    messages = [SystemMessage(content=_shop.SYSTEM_PROMPT), HumanMessage(content=args.question)]
    for step in range(1, 7):
        print(f"\n==== 第 {step} 轮 ====")
        reply = stream_reply(model, messages, include_usage=not args.without_usage)
        messages.append(reply)
        if not reply.tool_calls:
            return
        for call in reply.tool_calls:
            result = _shop.execute(call["name"], json.dumps(call["args"], ensure_ascii=False))
            print(f"  完整调用：{call['name']}({call['args']}) → {result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    print("达到 6 轮上限，本次未完成；请缩小查询范围。")


if __name__ == "__main__":
    main()

# 试试改：
# - 写代码：在 p13e 的 messages 事件中按运行 id 分组累计 AIMessageChunk，再核对 updates 里的完整调用。
# - 写代码：增加重复 id、缺 id 和 content_filter 结束的案例，确认不会执行工具。
