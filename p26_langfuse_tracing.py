# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "langchain>=1.4,<2",
#   "langchain-openai>=1.7,<2",
#   "langgraph>=1.2,<2",
#   "langfuse>=3.15,<4",
#   "python-dotenv>=1.1,<2",
#   "pydantic-settings>=2.10,<3",
# ]
# ///
"""资料库 26：用 Langfuse CallbackHandler 追踪 LangChain / LangGraph。

uv run p26_langfuse_tracing.py --upload
uv run p26_langfuse_tracing.py --check   只导出到内存，不连接 Langfuse/模型

同一个 trace 下观察任务根 span、图节点、LLM generation、工具执行、用量与错误。
--upload 会把固定的公开教学问答发送到你配置的 Langfuse 实例。
现有 .env 增加 LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY、LANGFUSE_BASE_URL 即可。
本例固定 SDK 3.x（需要支持此 SDK 的服务端）；不要混用新版 SDK 文档里的不同接口。
flush 只表示已请求刷新；之后尝试读回 trace，只有读回成功才报告“平台已可查询”。
"""

import argparse
import json
import re
import time
import uuid
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import Field

import _shop
from _langchain import make_model
from setting import Settings


class TracingSettings(Settings):
    langfuse_public_key: str = ""
    langfuse_secret_key: str = Field(default="", repr=False)
    langfuse_base_url: str = ""


def mask(data, **kwargs):
    """演示性脱敏：生产还需按数据分类裁剪输入，而不是仅靠正则。"""
    if isinstance(data, dict):
        return {
            k: "[REDACTED]"
            if k.lower() in {"api_key", "secret", "secret_key", "authorization", "phone", "address"}
            else mask(v)
            for k, v in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [mask(v) for v in data]
    if isinstance(data, str):
        data = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", data)
        data = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", data)
        return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[KEY]", data)
    return data


@tool
def lookup_demo_policy(topic: Literal["return", "shipping"]) -> str:
    """查询公开教学店铺政策：return 是退换货，shipping 是配送。没有订单或顾客信息。"""
    row = _shop.KNOWLEDGE[0 if topic == "return" else 1]
    return json.dumps({"source": row["id"], "text": row["text"]}, ensure_ascii=False)


class State(MessagesState):
    steps: int
    stopped: bool


def build_graph(model):
    def call(state: State):
        if state["steps"] >= 5:
            return {"stopped": True}
        reply = model.invoke(
            [
                SystemMessage(content="你是客服，政策回答必须先查公开政策工具，引用 source；不编造工具结果。"),
                *state["messages"],
            ]
        )
        return {"messages": [reply], "steps": state["steps"] + 1}

    graph = StateGraph(State)
    graph.add_node("model", call)
    graph.add_node("tools", ToolNode([lookup_demo_policy], handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", lambda s: END if s["stopped"] or not s["messages"][-1].tool_calls else "tools")
    graph.add_edge("tools", "model")
    return graph.compile()


def trace_task(client: Langfuse, graph, public_key: str):
    handler = CallbackHandler(public_key=public_key)  # 同一客户端对应的 project
    start = time.perf_counter()
    with client.start_as_current_observation(
        name="p26-public-policy",
        as_type="agent",
        input={"question": "质量问题退货运费谁出？"},
        metadata={"example": "p26", "data": "public-demo"},
    ) as root:
        trace_id = root.trace_id
        try:
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content="请查询质量问题退货运费由谁承担，引用政策来源。")],
                    "steps": 0,
                    "stopped": False,
                },
                {"callbacks": [handler], "recursion_limit": 20, "run_name": "policy-graph"},
            )
            if result["stopped"]:
                raise RuntimeError("模型轮数上限，任务未完成")
            text = str(result["messages"][-1].content)
            root.update(output={"answer": text}, metadata={"model_steps": result["steps"]})
            print(text)
        except Exception as exc:
            root.update(level="ERROR", status_message=type(exc).__name__, output={"status": "failed"})
            print("任务失败，根 span 标记 ERROR：", type(exc).__name__)
            raise
        finally:
            print(f"本地耗时={time.perf_counter() - start:.3f}s；trace_id={trace_id}")
    return trace_id


def check():
    from unittest.mock import patch

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert mask({"phone": "13800001111", "nested": ["x@example.org", "sk-offline-demo"]}) == {
        "phone": "[REDACTED]",
        "nested": ["[EMAIL]", "[KEY]"],
    }
    # 只在本地契约检查中替换导出传输层，Langfuse SDK、CallbackHandler 和图均真实运行。
    # 正常 --upload 入口无替代传输或模型。
    exporter = InMemorySpanExporter()
    public_key = "pk-check-" + uuid.uuid4().hex
    with patch("langfuse._client.span_processor.OTLPSpanExporter", return_value=exporter):
        client = Langfuse(
            public_key=public_key, secret_key="offline-contract-only", base_url="http://127.0.0.1:1", mask=mask
        )
        try:
            replies = iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "policy-1", "name": "lookup_demo_policy", "args": {"topic": "return"}}],
                        usage_metadata={"input_tokens": 30, "output_tokens": 8, "total_tokens": 38},
                    ),
                    AIMessage(
                        content="质量问题运费由小店承担。[policy/退换货政策]",
                        usage_metadata={"input_tokens": 60, "output_tokens": 12, "total_tokens": 72},
                    ),
                ]
            )
            trace_id = trace_task(client, build_graph(GenericFakeChatModel(messages=replies)), public_key)
            client.flush()
            spans = exporter.get_finished_spans()
            assert spans and all(f"{s.context.trace_id:032x}" == trace_id for s in spans)
            types = {s.attributes.get("langfuse.observation.type") for s in spans}
            assert {"agent", "generation", "tool"} <= types, types
            assert any(s.parent is not None for s in spans)
            assert any("usage" in str(k) for s in spans for k in s.attributes), "generation 用量未记录"
            print("本地实际 span：", [(s.name, s.attributes.get("langfuse.observation.type")) for s in spans])
            # 故障路径：失败的图执行也应结束根 span 并标记错误。
            from langchain_core.runnables import RunnableLambda

            def fail(_):
                raise RuntimeError("教学用本地故障")

            try:
                trace_task(client, RunnableLambda(fail), public_key)
            except RuntimeError:
                pass
            client.flush()
            assert any(s.attributes.get("langfuse.observation.level") == "ERROR" for s in exporter.get_finished_spans())
        finally:
            client.shutdown()
    print("Langfuse 实际根 span、图/模型/工具子 span、用量、故障与脱敏检查通过；仅内存导出，没有上传。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="允许上传固定公开教学数据的追踪")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if not args.upload:
        parser.error("真实追踪请显式使用 --upload；本地检查用 --check")
    config = TracingSettings()
    if not all((config.langfuse_public_key, config.langfuse_secret_key, config.langfuse_base_url)):
        raise RuntimeError("缺少 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL，请在现有 .env 配置")
    model = make_model().bind_tools([lookup_demo_policy])
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
        mask=mask,
        timeout=10,
    )
    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 凭据验证失败")
        print("开始上传公开教学数据的 LangGraph 追踪。")
        trace_id = trace_task(client, build_graph(model), config.langfuse_public_key)
        client.flush()
        print("已请求 flush；trace 链接：", client.get_trace_url(trace_id=trace_id))
        try:
            record = client.api.trace.get(trace_id)
            print("平台已可查询：", record.id)
        except Exception as exc:
            print("暂未确认平台可查询（可能尚在入库或读取失败）：", type(exc).__name__)
            print("请稍后按 trace_id 查看；这不等于上传已成功。")
    finally:
        client.flush()
        client.shutdown()


if __name__ == "__main__":
    main()

# 试试改：
# - 写代码：将相同 CallbackHandler 接到 p19 的 config.callbacks，检查审批恢复后 trace 的关联。
# - 写代码：为 p18a 增加 retriever span，记录候选数量与排名，不上传原始敏感资料。
