# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "langchain-openai>=1.7,<2",
#   "langgraph>=1.2,<2",
#   "python-dotenv>=1.1,<2",
#   "pydantic-settings>=2.10,<3",
# ]
# ///
"""资料库 19：把工程机制组装进同一个 LangGraph 工具循环。

uv run p19_mini_harness.py
uv run p19_mini_harness.py --decision reject
uv run p19_mini_harness.py --check   检查真实图的暂停/恢复等边界，不调用模型

before_model → model → validate → approval → execute → after_tool → before_model。
validate 先校验整批调用；审批节点用 interrupt 暂停，Command 恢复，随后才执行动作。
高风险工具仅在本次状态中创建退款草稿，不退款、不写外部服务。非交互运行默认拒绝。
--decision approve 是教学用预置审批决定，生产必须接真实审批人及身份验证。
预算使用 UTF-8 字节近似 token 数（包括工具 Schema），并预留输出；不是供应商精确计费。
"""

import argparse
import hashlib
import json
import math
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import Field, ValidationError

import _shop
from _langchain import SHOP_TOOLS, make_model


class RefundArgs(_shop.OrderArgs):
    amount: float = Field(gt=0, le=10000, allow_inf_nan=False, description="退款草稿金额，不能超过订单实付")
    reason: str = Field(min_length=2, max_length=100, description="创建草稿的原因")


@tool(args_schema=RefundArgs)
def create_refund_draft(order_id: str, amount: float, reason: str) -> str:
    """创建当前顾客的本地退款草稿，必须逐次审批；不会退款或向商家提交。"""
    order = json.loads(_shop.execute("query_order", json.dumps({"order_id": order_id})))
    if "error" in order:
        return json.dumps(order, ensure_ascii=False)
    if amount > order["amount"]:
        return json.dumps({"error": "草稿金额超过订单实付"}, ensure_ascii=False)
    return json.dumps(
        {"status": "local_draft_only", "order_id": order_id, "amount": amount, "reason": reason}, ensure_ascii=False
    )


TOOLS = {t.name: t for t in [*SHOP_TOOLS, create_refund_draft]}
PROMPT = _shop.SYSTEM_PROMPT + "\n可按用户要求创建本地退款草稿，但必须说明草稿不是退款或提交成功。"


class State(MessagesState):
    steps: int
    tools_used: int
    input_used: int
    failures: dict
    prepared: list
    cursor: int
    approved_key: str
    outcome: dict
    drafts: dict
    stop: str


def fingerprint(name: str, args: dict) -> str:
    return hashlib.sha256(
        json.dumps([name, args], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def estimate(messages: list, schemas: list) -> int:
    data = json.dumps({"messages": [m.model_dump() for m in messages], "tools": schemas}, ensure_ascii=False)
    return math.ceil(len(data.encode()) / 3)


def build_graph(model, *, context_limit=6000, input_limit=20000, max_steps=6, max_tools=8):
    schemas = [
        t.args_schema.model_json_schema() | {"name": name, "description": t.description} for name, t in TOOLS.items()
    ]

    def before_model(state: State):
        cost = estimate([SystemMessage(content=PROMPT), *state["messages"]], schemas)
        reason = ""
        if state["steps"] >= max_steps:
            reason = "模型轮数预算耗尽"
        elif cost + 512 > context_limit:
            reason = "上下文预算不足（已预留 512 输出 tokens），需要压缩或缩小输入"
        elif state["input_used"] + cost > input_limit:
            reason = "累计输入预算不足"
        print(f"[before_model] steps={state['steps']} input≈{cost} cumulative≈{state['input_used']}")
        return {"stop": reason, "input_used": state["input_used"] + (0 if reason else cost)}

    def call_model(state: State):
        reply = model.invoke([SystemMessage(content=PROMPT), *state["messages"]])
        if reply.invalid_tool_calls:
            return {"stop": "模型工具参数无法解析，未执行工具", "steps": state["steps"] + 1}
        return {"messages": [reply], "steps": state["steps"] + 1}

    def validate(state: State):
        calls = state["messages"][-1].tool_calls
        prior_ids = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
        prepared, error = [], ""
        try:
            ids = [c["id"] for c in calls]
            if any(not i or i in prior_ids for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("调用 id 缺失、重复或已执行")
            if state["tools_used"] + len(calls) > max_tools:
                raise ValueError("工具调用预算不足，整批未执行")
            for call in calls:
                if call["name"] not in TOOLS:
                    raise ValueError(f"未授权工具 {call['name']}")
                args = TOOLS[call["name"]].args_schema.model_validate(call["args"]).model_dump()
                prepared.append({**call, "args": args, "key": fingerprint(call["name"], args)})
        except (ValueError, ValidationError) as exc:
            error = f"参数/调用校验失败，整批未执行：{exc}"
        if error:
            # 没有执行前面的合法项；每个调用都回填结果，避免下一轮遗留孤立消息。
            return {
                "messages": [
                    ToolMessage(content=json.dumps({"error": error}, ensure_ascii=False), tool_call_id=c["id"])
                    for c in calls
                ],
                "stop": error,
            }
        print(f"[validate] 整批 {len(prepared)} 项通过")
        return {"prepared": prepared, "cursor": 0, "approved_key": ""}

    def approval(state: State):
        call = state["prepared"][state["cursor"]]
        if call["name"] == "create_refund_draft":
            # interrupt 之前不能执行动作：恢复时本节点会重新开始。
            decision = interrupt(
                {
                    "tool": call["name"],
                    "args": call["args"],
                    "fingerprint": call["key"],
                    "notice": "只创建本地退款草稿；审批仅适用于以上参数。",
                }
            )
            if (
                not isinstance(decision, dict)
                or decision.get("approved") is not True
                or decision.get("fingerprint") != call["key"]
            ):
                remaining = state["prepared"][state["cursor"] :]
                return {
                    "stop": "审批被拒绝或参数指纹不匹配，未创建草稿",
                    "messages": [
                        ToolMessage(content='{"error":"审批拒绝，本项未执行"}', tool_call_id=c["id"]) for c in remaining
                    ],
                }
        return {"approved_key": call["key"]}

    def execute(state: State):
        call = state["prepared"][state["cursor"]]
        if state["approved_key"] != fingerprint(call["name"], call["args"]):
            raise ValueError("执行参数与审批参数不一致")
        result = json.loads(TOOLS[call["name"]].invoke(call["args"]))
        drafts = dict(state["drafts"])
        if call["name"] == "create_refund_draft" and "error" not in result:
            drafts[call["id"]] = result  # 唯一的“写入”是可检查点化的本地状态
        print(f"[execute] {call['name']} → {result}")
        return {
            "messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"])],
            "outcome": {"key": call["key"], "error": isinstance(result, dict) and "error" in result},
            "tools_used": state["tools_used"] + 1,
            "drafts": drafts,
            "approved_key": "",
        }

    def after_tool(state: State):
        failures = dict(state["failures"])
        key, failed = state["outcome"]["key"], state["outcome"]["error"]
        failures[key] = failures.get(key, 0) + 1 if failed else 0
        cursor = state["cursor"] + 1
        print(f"[after_tool] 同参数连续失败={failures[key]}")
        if failures[key] >= 2:
            remaining = state["prepared"][cursor:]
            return {
                "failures": failures,
                "cursor": cursor,
                "stop": "同参数工具重复失败两次，硬停止；未执行剩余调用",
                "messages": [
                    ToolMessage(content='{"error":"重复失败保护，本项未执行"}', tool_call_id=c["id"]) for c in remaining
                ],
            }
        return {"failures": failures, "cursor": cursor}

    graph = StateGraph(State)
    for name, node in [
        ("before_model", before_model),
        ("model", call_model),
        ("validate", validate),
        ("approval", approval),
        ("execute", execute),
        ("after_tool", after_tool),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "before_model")
    graph.add_conditional_edges("before_model", lambda s: END if s["stop"] else "model")
    graph.add_conditional_edges("model", lambda s: END if s["stop"] or not s["messages"][-1].tool_calls else "validate")
    graph.add_conditional_edges("validate", lambda s: END if s["stop"] else "approval")
    graph.add_conditional_edges("approval", lambda s: END if s["stop"] else "execute")
    graph.add_edge("execute", "after_tool")
    graph.add_conditional_edges(
        "after_tool",
        lambda s: END if s["stop"] else ("approval" if s["cursor"] < len(s["prepared"]) else "before_model"),
    )
    return graph.compile(checkpointer=InMemorySaver())


def initial_state(question: str) -> State:
    return {
        "messages": [HumanMessage(content=question)],
        "steps": 0,
        "tools_used": 0,
        "input_used": 0,
        "failures": {},
        "prepared": [],
        "cursor": 0,
        "approved_key": "",
        "outcome": {},
        "drafts": {},
        "stop": "",
    }


def check():
    from langchain_core.runnables import RunnableLambda

    def call(name, args, call_id):
        return {"name": name, "args": args, "id": call_id}

    def run(replies, **options):
        # 离线状态机测试专用的消息输入；正式入口不提供替代模型模式。
        sequence = iter(replies)
        graph = build_graph(RunnableLambda(lambda _: next(sequence)), **options)
        config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 80}
        return graph, config, graph.invoke(initial_state("测试"), config)

    draft = call("create_refund_draft", {"order_id": "ORD-20261002", "amount": 10, "reason": "质量问题"}, "draft-1")
    for approved, key_valid in [(False, True), (True, False), (True, True)]:
        graph, config, state = run([AIMessage(content="", tool_calls=[draft]), AIMessage(content="仅创建本地草稿。")])
        request = state["__interrupt__"][0].value
        assert not state["drafts"] and state["tools_used"] == 0
        result = graph.invoke(
            Command(resume={"approved": approved, "fingerprint": request["fingerprint"] if key_valid else "wrong"}),
            config,
        )
        assert len(result["drafts"]) == (1 if approved and key_valid else 0)
        assert result["tools_used"] == (1 if approved and key_valid else 0)
        if approved and key_valid:
            # 读取同一检查点不会重放动作；生产外部写入仍需要服务端幂等键。
            assert len(graph.get_state(config).values["drafts"]) == 1
    good = call("query_order", {"order_id": "ORD-20261001"}, "good")
    bad = call("query_order", {"order_id": "bad", "user_id": "u2"}, "bad")
    _, _, result = run([AIMessage(content="", tool_calls=[good, bad])])
    assert result["tools_used"] == 0 and result["stop"].startswith("参数/")
    missing = {"order_id": "ORD-99999999"}
    _, _, result = run(
        [AIMessage(content="", tool_calls=[call("query_order", missing, "a"), call("query_order", missing, "b"), good])]
    )
    assert result["tools_used"] == 2 and "重复失败" in result["stop"]
    assert result["messages"][-1].tool_call_id == "good"
    _, _, result = run([], context_limit=1)
    assert result["steps"] == 0 and "上下文预算" in result["stop"]
    _, _, result = run([], input_limit=1)
    assert result["steps"] == 0 and "累计输入" in result["stop"]
    _, _, result = run([AIMessage(content="", tool_calls=[good])], max_tools=0)
    assert result["tools_used"] == 0 and "工具调用预算" in result["stop"]
    _, _, result = run([AIMessage(content="", tool_calls=[good])], max_steps=1)
    assert result["tools_used"] == 1 and "模型轮数" in result["stop"]
    assert fingerprint("x", {"a": 1, "b": 2}) == fingerprint("x", {"b": 2, "a": 1})
    print("LangGraph 审批暂停/恢复/参数绑定、整批校验、重复失败硬停与四种预算检查通过；未调用模型。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="请查 ORD-20261002，因质量问题为它创建 10 元的退款草稿。")
    parser.add_argument("--decision", choices=["ask", "approve", "reject"], default="ask")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    graph = build_graph(make_model().bind_tools(list(TOOLS.values())).bind(max_tokens=512))
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 80}
    result = graph.invoke(initial_state(args.question), config)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        print("待审批：", json.dumps(request, ensure_ascii=False, indent=2))
        decision = args.decision
        if decision == "ask":
            decision = (
                "approve"
                if sys.stdin.isatty() and input("仅允许以上参数创建本地草稿？输入 yes：").strip() == "yes"
                else "reject"
            )
        else:
            print("教学预置审批决定：", decision)
        result = graph.invoke(
            Command(resume={"approved": decision == "approve", "fingerprint": request["fingerprint"]}), config
        )
    print("停止原因：" + result["stop"] if result["stop"] else result["messages"][-1].content)
    print(f"模型轮数={result['steps']}，工具次数={result['tools_used']}，本地草稿={result['drafts']}")


if __name__ == "__main__":
    main()

# 试试改：
# - 写代码：将 p16b 压缩接到 before_model；压缩调用也要计入累计预算，不能无限借压缩绕过限额。
# - 写代码：把草稿接到本地 SQLite，以调用 id 作为幂等键；模拟节点重试，验证不会重复写入。
