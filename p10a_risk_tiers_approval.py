# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=3.0", "pydantic>=2.11", "python-dotenv>=1.1,<2", "pydantic-settings>=2.10,<3"]
# ///
"""资料库 10《工具执行与沙箱》：按风险给工具分级，高风险的要人确认，每次调用都留审计记录。

uv run p10a_risk_tiers_approval.py             在终端里运行会停下来问你批不批准
uv run p10a_risk_tiers_approval.py --approve   不问，全部批准（给脚本批量跑用）

看什么：
1. 低风险（只读）直接执行；中风险（草稿）执行并记录；高风险（真的退款）必须人确认；没登记的工具按高风险处理
2. 确认界面展示的是真实的操作和参数，不是模型写的一段说明（资料库 27 篇：别让模型的话术说服人）
3. 批准只对这一组参数有效：参数一变就要重新确认
4. 审计日志：谁、什么时候、调了什么、结果如何
"""

import json
import sys
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import _shop
from _llm import LLM

RISK = {
    "query_order": "低",
    "track_shipment": "低",
    "create_refund_draft": "中",
    "submit_refund": "高",
}
AUDIT: list[dict] = []
APPROVED: set[tuple[str, str]] = set()  # 已批准的（工具名，参数）组合


class DraftArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 多传的字段直接报错，免得模型以为传了就生效

    order_id: str = Field(pattern=r"^ORD-\d{8}$", description="订单号")
    amount: float = Field(gt=0, description="退款金额，单位元")


class SubmitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str = Field(description="退款草稿编号，由 create_refund_draft 返回")


TOOLS = _shop.schemas(["query_order", "track_shipment"]) + [
    {"type": "function", "function": {"name": "create_refund_draft", "description": "生成退款申请草稿，不会真的退款",
                                      "parameters": DraftArgs.model_json_schema()}},
    {"type": "function", "function": {"name": "submit_refund", "description": "提交退款草稿，真的发起退款",
                                      "parameters": SubmitArgs.model_json_schema()}},
]


def create_refund_draft(a: DraftArgs) -> dict:
    return {"draft_id": "RF-0001", "order_id": a.order_id, "amount": a.amount, "status": "草稿，未提交"}


def submit_refund(a: SubmitArgs) -> dict:
    return {"draft_id": a.draft_id, "status": "已提交退款（演示，没有真的退钱）"}


LOCAL_ARGS = {"create_refund_draft": DraftArgs, "submit_refund": SubmitArgs}  # 店铺工具的参数由 _shop.execute 校验


def ask_human(name: str, arguments: str) -> bool:
    print(f"\n  ┌ 需要确认的高风险操作\n  │ 工具：{name}\n  │ 参数：{arguments}\n  │ 影响：会真的发起退款，不能撤销")
    if "--approve" in sys.argv:
        print("  └ --approve：自动批准")
        return True
    if not sys.stdin.isatty():
        print("  └ 不是在终端里运行，没人能确认，默认拒绝")
        return False
    return input("  └ 批准吗？输入 y 批准：").strip().lower() == "y"


def audit(name: str, arguments: str, level: str, result: str) -> None:
    AUDIT.append({"at": datetime.now(), "tool": name, "args": arguments, "level": level, "result": result})


def guarded_execute(name: str, arguments: str) -> str:
    level = RISK.get(name, "高")  # 没登记的工具一律按高风险
    args = None
    if name in LOCAL_ARGS:
        try:  # 先校验：参数不合法就直接打回给模型，不拿去给人审批，也不执行
            args = LOCAL_ARGS[name].model_validate_json(arguments or "{}")
        except ValidationError as exc:
            audit(name, arguments, level, "参数不合法")
            return _shop.args_error(exc)
    if level == "高" and (name, arguments) not in APPROVED:
        if not ask_human(name, arguments):
            result = {"error": "用户没有批准这个操作，请告诉顾客需要人工确认后才能办理"}
            audit(name, arguments, level, "拒绝")
            return json.dumps(result, ensure_ascii=False)
        APPROVED.add((name, arguments))  # 只批准这一组参数；参数变了要重新问
    if isinstance(args, DraftArgs):
        result = create_refund_draft(args)
    elif isinstance(args, SubmitArgs):
        result = submit_refund(args)
    else:
        result = json.loads(_shop.execute(name, arguments))
    audit(name, arguments, level, "执行")
    return json.dumps(result, ensure_ascii=False)


llm = LLM()
print(llm.describe())
print("==== 1. 顾客：耳机有质量问题，帮我把 ORD-20261001 退了 ====")
messages = [
    {"role": "system", "content": _shop.SYSTEM_PROMPT},
    {"role": "user", "content": "耳机有质量问题，帮我把 ORD-20261001 退了"},
]
for _ in range(5):
    reply = llm.chat(messages, TOOLS)
    messages.append(reply.as_message())
    if not reply.tool_calls:
        print(f"\n  回复顾客：{reply.content}")
        break
    for c in reply.tool_calls:
        result = guarded_execute(c.name, c.arguments)
        messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        print(f"  [{RISK.get(c.name, '高')}风险] {c.name}({c.arguments}) → {result}")

print("\n==== 2. 批准只对同一组参数有效 ====")
changed = '{"draft_id": "RF-9999"}'
print(f"  同一个工具换了参数 {changed}：{'需要重新确认' if ('submit_refund', changed) not in APPROVED else '已批准'}")

print("\n==== 3. 审计日志 ====")
for i, a in enumerate(AUDIT, 1):
    who = f"顾客 {_shop.CURRENT_USER}"
    print(f"  {i}. {a['at']:%H:%M:%S} {who} {a['level']}风险 {a['tool']} {a['args']} → {a['result']}")

# 试试改：
# - 在终端里运行，分别输入 y 和 n，看模型拿到“没批准”之后怎么回复
# - 把 submit_refund 从 RISK 里删掉，看没登记的工具是不是按高风险处理
# - 写代码：写一个回归案例：批准后修改退款金额，确认原批准记录无法授权新参数。
