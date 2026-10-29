"""资料库 13 第 8 节：Deep Agents Harness 的小店订单分析示例。

uv run p13i_deepagents_harness.py
uv run p13i_deepagents_harness.py --decision reject

看什么：create_deep_agent、task 子 Agent、虚拟文件、Skills / Memory、可选规划、工具审批。
需要支持 tool calling 的真实模型。仅访问 _shop 演示数据；文件在 StateBackend 中，不落本机磁盘。
审批值由命令行预设以方便课堂复现；生产系统应等待真实审核人决定。
"""

import argparse

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from _langchain import SHOP_TOOLS, make_model

REPORT_PATH = "/reports/order.md"
SKILL = """---
name: order-report
description: 将订单核查结果写成有依据的中文客服报告。
---
报告分为订单信息、物流现状、建议三段；事实必须来自工具结果。
未知信息明确标注未知，不承诺未确认的到货时间，不执行退款或联系客户。
"""


def build_agent(model):
    return create_deep_agent(
        model=model,
        tools=SHOP_TOOLS,
        backend=StateBackend(),  # 0.7.32 的实例接口；虚拟路径不是本机绝对路径。
        checkpointer=InMemorySaver(),
        middleware=[TodoListMiddleware()],  # 文档所述：0.7 起规划不再默认启用，需要显式添加。
        memory=["/memory/AGENTS.md"],
        skills=["/skills/"],
        subagents=[{
            "name": "order_researcher",
            "description": "核对订单商品、金额和物流，返回精炼事实及所用工具名。",
            "system_prompt": "你是订单核查员。必须用工具查证，只返回事实和来源，不写文件，不承诺退款。",
            "model": model,
            "tools": SHOP_TOOLS,
        }],
        interrupt_on={"write_file": {"allowed_decisions": ["approve", "reject"]}},
        system_prompt=(
            "你负责生成订单报告。用 write_todos 规划，调用 order_researcher 子 Agent 核查订单，"
            "按 order-report 技能写报告到 /reports/order.md，再读取报告确认，最后给出中文摘要。"
            "文件写入被拒绝后不要再次尝试写入，直接给出文字结论并说明未保存。"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", choices=["approve", "reject"], default="approve")
    args = parser.parse_args()
    agent = build_agent(make_model())
    config = {"configurable": {"thread_id": "order-report-u1"}, "recursion_limit": 60}

    print("==== 1. 预置工作区指令与技能，由主 Agent 委派订单核查 ====")
    result = agent.invoke({
        "messages": [("user", "请核查 ORD-20261001 的商品、金额与物流，生成报告。")],
        "files": {
            "/memory/AGENTS.md": create_file_data("用中文回答。顾客身份由工具绑定，不请求身份证或电话。"),
            "/skills/order-report/SKILL.md": create_file_data(SKILL),
        },
    }, config)

    print("\n==== 2. 写文件前暂停，审批后由同一 thread 恢复 ====")
    for _ in range(3):  # 防止模型被拒绝后反复请求审批；这只是教学上限。
        interrupts = result.get("__interrupt__", ())
        if not interrupts:
            break
        resumes = {}
        for pending in interrupts:
            requests = pending.value["action_requests"]
            for request in requests:
                print(f"  待审批：{request['name']}({request['args']})")
            print(f"  教学预设决定：{args.decision}（没有真实人工点击）")
            resumes[pending.id] = {"decisions": [{"type": args.decision} for _ in requests]}
        result = agent.invoke(Command(resume=resumes), config)
    if result.get("__interrupt__"):
        raise SystemExit("审批请求超过教学上限，停止演示；请检查模型是否遵守拒绝决定")

    print("\n==== 3. 主 Agent 看到的是子任务结果，文件保存在图状态中 ====")
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []):
            print(f"  主 Agent 调用：{call['name']}")
    print(result["messages"][-1].content)
    print(f"  任务清单：{result.get('todos', [])}")
    files = agent.get_state(config).values.get("files", {})
    print(f"  虚拟文件：{list(files)}")
    if REPORT_PATH in files:
        print("\n".join(files[REPORT_PATH]["content"]))
    else:
        print("  没有生成报告文件：可能被拒绝，或模型没有按要求调用写文件工具。")


if __name__ == "__main__":
    main()

# 试试改：去掉 TodoListMiddleware 对比工具列表；用 --decision reject 看文件是否生成。
# 短任务未必触发自动总结 / 大结果卸载；不能因为安装了 middleware 就声称本轮已触发这些机制。
# - 写代码：为报告内容编写校验函数，要求包含订单号与工具来源，缺项时明确显示未通过验收。
