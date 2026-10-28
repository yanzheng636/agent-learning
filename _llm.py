"""示例共用的真实模型调用，通过 setting.settings 读取模型配置。

编辑本目录的 .env 后直接 uv run 示例文件；已存在的环境变量优先。
LLM_BASE_URL 留空时使用 OpenAI 官方地址。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from setting import settings

FALLBACK_REPLY = "抱歉，这个问题我暂时没处理好，已为您转接人工客服。"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 和接口里一样是 JSON 字符串；模型可能给出不合法的 JSON，由执行工具的一方校验

    def args(self) -> dict[str, Any]:
        return json.loads(self.arguments or "{}")


@dataclass
class Reply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # 命中 Prompt 缓存的输入 token

    def as_message(self) -> dict[str, Any]:
        """模型这一步的输出要原样放回对话历史，下一步模型才知道自己调过什么。"""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


Messages = list[dict[str, Any]]


def tool_results(messages: Messages) -> list[dict[str, Any]]:
    """提取最近一条用户消息之后的工具调用记录，供评估检查过程。"""
    start = max(i for i, m in enumerate(messages) if m["role"] == "user")
    called: dict[str, tuple[str, str]] = {}
    out = []
    for m in messages[start:]:
        for c in m.get("tool_calls") or []:
            called[c["id"]] = (c["function"]["name"], c["function"]["arguments"])
        if m["role"] == "tool":
            name, arguments = called[m["tool_call_id"]]
            try:
                result = json.loads(m["content"])
            except json.JSONDecodeError:
                result = m["content"]
            out.append({"name": name, "args": json.loads(arguments or "{}"), "result": result})
    return out


def count_tokens(obj: Any) -> int:
    """数 token。装了 tiktoken 用 OpenAI 的 o200k_base 编码，没装就粗估：汉字算 1 个，其他字符 4 个算 1 个。

    不同模型的分词器不同，这里的数只用来比较大小。
    """
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    try:
        import tiktoken
    except ImportError:
        cjk = sum(1 for ch in s if "一" <= ch <= "鿿")
        return cjk + (len(s) - cjk) // 4
    return len(tiktoken.get_encoding("o200k_base").encode(s))


class LLM:
    def __init__(self) -> None:
        if not settings.llm_api_key or not settings.llm_model:
            raise RuntimeError("请在 .env 或环境变量中设置 LLM_API_KEY 和 LLM_MODEL；第三方服务还需设置 LLM_BASE_URL")
        from openai import OpenAI

        self.model = settings.llm_model
        # 空字符串也当作没设置，交给 SDK 用默认地址
        self._client = OpenAI(
            base_url=settings.llm_base_url or None, api_key=settings.llm_api_key, timeout=60,
        )
        self.calls = 0
        self.prompt_tokens = 0  # 累计的输入 token，用来比较不同做法的花费

    def describe(self) -> str:
        return f"模型：{self.model}（{settings.llm_base_url or 'OpenAI 官方地址'}）"

    def chat(self, messages: Messages, tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Reply:
        self.calls += 1
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        msg = resp.choices[0].message
        usage = resp.usage
        details = usage.prompt_tokens_details if usage else None
        reply = Reply(
            content=msg.content,
            tool_calls=[ToolCall(c.id, c.function.name, c.function.arguments) for c in msg.tool_calls or []],
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cached_tokens=(details.cached_tokens or 0) if details else 0,
        )
        self.prompt_tokens += reply.prompt_tokens
        return reply


def run_agent(
    llm: LLM,
    messages: Messages,
    tools: list[dict[str, Any]],
    execute: Callable[[str, str], str],
    *,
    max_steps: int = 5,
    label: str = "",
) -> str:
    """最小的 Agent 循环，和 p07a_react_loop.py 里手写的是同一个东西。messages 会被原地追加。"""
    for _ in range(max_steps):
        reply = llm.chat(messages, tools)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            return reply.content or ""
        for c in reply.tool_calls:
            result = execute(c.name, c.arguments)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            print(f"  {label}调用 {c.name}({c.arguments}) → {result[:80]}")
    return FALLBACK_REPLY


