# AI Agent 技术资料库 · 示例代码

这是一套面向 Python 开发者的 Agent 机制教程，配合飞书「AI Agent 技术资料库」学习。业务场景统一为**小店客服助手**：查订单、查物流、回答政策、处理售后和生成报告。

每个示例围绕一个机制展开，包含中文说明、运行输出和“试试改”练习。建议先运行观察，再阅读代码，最后修改参数对比结果。

> 文件名中的 `p04`、`p13` 对应资料库章节编号。本目录只包含已经实现的配套示例，编号不连续；下方按现有代码说明各章内容。

## 1. 环境准备

### 1.1 安装 uv

官方安装文档：[uv Installation](https://docs.astral.sh/uv/getting-started/installation/)。

如果已经安装了 Python，也可以通过 pip 安装。macOS / Linux：

```bash
python3 -m pip install --user uv
```

以上安装方式任选一种。安装后确认 `uv` 在 PATH 中，以下命令在 PowerShell 和 macOS / Linux 终端中都可使用：

```bash
uv --version
```

**下文命令都在本 README 所在的 `agent-learning` 目录执行。** 请先在终端切换到该目录。

### 1.2 一步配置环境：uv sync

```bash
uv sync
```

这一条命令完成全部准备：

1. 按 [`.python-version`](.python-version) 准备 Python 3.12，本机没有时 uv 会自动下载，不需要另外安装 Python；
2. 在本目录创建虚拟环境 `.venv`；
3. 按 [`uv.lock`](uv.lock) 安装 [`pyproject.toml`](pyproject.toml) 声明的全部依赖，包括检索、追踪示例用的包和代码检查工具 ruff。锁文件固定了每个包的版本，所有人装到的环境一致。

首次安装要下载一百多个包，`.venv` 约 500 MB，耗时主要取决于网络；装好后再运行示例不会重复下载。

之后照常用 `uv run 文件名.py` 运行示例，uv 直接使用这个 `.venv`；`pyproject.toml` 或 `uv.lock` 有变化时，`uv run` 会先自动同步。在 PyCharm / VS Code 里把解释器设为 `.venv/bin/python`（Windows 为 `.venv\Scripts\python.exe`），就可以直接点运行按钮。

依赖统一写在 `pyproject.toml`，示例文件里不再单独声明。需要新增依赖时运行 `uv add 包名`，它会同时更新 `pyproject.toml` 和 `uv.lock`。

### 1.3 先验证无模型示例

还没有 API Key，也可以先运行：

```bash
uv run p13a_langgraph_basics.py
```

这个示例用普通 Python 函数演示状态、分支和审批。也可以一次运行五个无模型的框架示例：

```bash
bash run_frameworks.sh --offline
```

`--offline` 指**不调用模型**，不等于完全断网：使用 tiktoken 的其他示例首次运行还可能下载编码文件。

## 2. 配置模型：.env → Settings → 调用方

### 2.1 填写 .env

本目录的 [`.env.example`](.env.example) 是配置模板。已有 `.env` 就直接编辑；没有时创建一份，下面的命令不会覆盖已有文件：

```bash
cp -n .env.example .env
```

在 `.env` 中填写自己的模型信息，例如：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_API_KEY=替换成你的真实API密钥
```

| 环境变量           | Settings 属性             | 含义            | 是否必填               |
|----------------|-------------------------|---------------|--------------------|
| `LLM_API_KEY`  | `settings.llm_api_key`  | 模型服务的 API Key | 调用模型时必填            |
| `LLM_MODEL`    | `settings.llm_model`    | 供应商支持的模型名称    | 调用模型时必填            |
| `LLM_BASE_URL` | `settings.llm_base_url` | OpenAI 兼容接口地址 | 可留空，使用 OpenAI 官方地址 |

使用其他供应商时，同时替换地址、模型名和密钥；地址是否需要 `/v1` 以供应商要求为准。不要把具体的 `/chat/completions` 接口路径写进 base URL。

涉及工具的示例需要模型支持 **tool calling**；LangChain 链和 LangGraph 流式示例还需要相应的流式能力。“兼容 OpenAI 接口”并不保证这些能力全部可用。

`.env` 已加入 `.gitignore`，不要提交或在录屏中展示真实密钥；`.env.example` 只保留模板值。

### 2.2 配置统一由 setting.py 管理

[`setting.py`](setting.py) 定义了配置类，并在模块中创建共享实例：

```python
from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv() or None,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    llm_api_key: str = Field(default="", repr=False)
    llm_model: str = ""
    llm_base_url: str = ""


settings = Settings()
```

调用方只需要导入实例：

```python
from setting import settings

model_name = settings.llm_model
base_url = settings.llm_base_url
```

[`_llm.py`](_llm.py) 和 [`_langchain.py`](_langchain.py) 已使用这套配置，业务代码不用再分别调用 `os.getenv()`、`load_dotenv()` 或手动加载 shell 文件。

配置行为：

- 普通脚本运行时，`find_dotenv()` 从配置模块所在目录向上寻找 `.env`。
- 优先级为：显式传给 `Settings(...)` 的参数 → 非空环境变量 → `.env` → 字段默认值。
- 字段名不区分大小写，`LLM_MODEL` 对应 `llm_model`；空字符串配置会被忽略。
- Pydantic 按字段类型转换并校验值；`.env` 中未声明为字段的变量会被忽略。
- `BaseSettings` 直接读取 `.env`，不会把文件内容写入 `os.environ`，不需要再调用 `load_dotenv()`。
- `settings` 在导入时创建。修改 `.env` 后重启示例；需要独立实例时可以自行 `Settings()`。
- 模型字段默认留空，方便无模型示例导入公共模块；真正创建模型客户端时才检查密钥和模型名。

### 2.3 以后如何新增环境变量

例如要增加调试开关和请求超时，先在 `Settings` 类中添加：

```python
app_debug: bool = False
request_timeout: float = Field(default=60, gt=0)
```

再在 `.env` 中配置，并将对应模板补入 `.env.example`：

```dotenv
APP_DEBUG=true
REQUEST_TIMEOUT=30
```

调用时直接读取：

```python
from setting import settings

if settings.app_debug:
    print("调试模式已开启")

timeout = settings.request_timeout
```

这是扩展示例，当前配置类尚未添加这两个字段。新增字段后，还需要把值接到实际调用处，例如传给客户端的 `timeout` 参数；仅声明变量不会自动改变模型行为。

### 2.4 运行第一个真实模型示例

保存 `.env` 后直接运行：

```bash
uv run p04b_tool_call_roundtrip.py
```

观察模型提出工具请求、程序执行工具、工具结果回填、模型生成最终回答的完整过程。

旧的 `llm_env.sh` 仅保留兼容用途。新配置方式不需要它；若仍通过它启动，旧脚本导出的环境变量会优先于 `.env`。

## 3. 学习路线

| 阶段     | 推荐章节顺序                                 | 学习目标                       |
|--------|----------------------------------------|----------------------------|
| 基础机制   | 03 → 04 → 05 → 07 → 08 → 09 → 10 → 11       | 理解模型、工具、循环、控制流和 MCP 如何协作   |
| 框架实践   | 13：LangChain → LangGraph → Deep Agents | 将手写机制映射到框架 API，掌握状态和恢复     |
| 上下文与记忆 | 16 → 17 → 18 → 19 → 22                           | 控制上下文体积，管理记忆，理解子 Agent 的代价 |
| 可靠性与优化 | 24 → 25 → 26 → 27 → 28 → 29                 | 防循环、做评估、拦注入、处理故障并降低延迟      |

每章都可以单独运行。建议每次回答三个问题：**输入是什么、由谁决定下一步、执行结果存在哪里？**

## 4. 每一章讲什么

下表“模型”表示是否调用真实模型服务。标为“否”的示例不需要 API Key；标为“是”的示例会产生模型调用费用。

| 章节                               | 主要内容与观察点                                                  | 对应代码                                                                                  | 模型    |
|----------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------|-------|
| **03 · 提示工程** | LangChain 分层消息模板、partial 固定约束、缺变量和空输入提前报错 | [p03a_prompt_layers.py](p03a_prompt_layers.py) | 是（可仅渲染） |
| **04 · 结构化输出与 Function Calling** | 工具定义如何发给模型；模型只提出调用请求；执行后用 `tool_call_id` 对齐工具结果，再请求最终回答   | [p04b_tool_call_roundtrip.py](p04b_tool_call_roundtrip.py)                            | 是     |
| **05 · Token、成本与调用约束**           | 中英文 token 对比；提示、工具和历史分别占多少；多轮 Agent 的累计输入成本               | [p05a_count_tokens.py](p05a_count_tokens.py)                                          | 否     |
| **05 · 流式交互** | 首事件 / 首文本 / 首工具延迟；LangChain 合并分片，严格解析完整 JSON 后执行 | [p05b_streaming.py](p05b_streaming.py) | 是 |
| **07 · Agent 循环**                | ReAct 每步决策与步数上限；Plan-and-Execute 先生成计划，再按依赖分批并发执行，最后汇总    | [p07a_react_loop.py](p07a_react_loop.py)、[p07c_plan_execute.py](p07c_plan_execute.py) | 是     |
| **08 · 常用编排模式**                  | 固定链式流程“抽取 → 业务关卡 → 回复”；按问题类别路由；区分代码控制流程与模型自主选动作           | [p08a_chain_router.py](p08a_chain_router.py)                                          | 是     |
| **09 · 工具设计**                    | Pydantic 生成参数 Schema；处理缺字段、多字段、格式错误；把错误回填给模型；压缩工具结果       | [p09a_tool_schema_pydantic.py](p09a_tool_schema_pydantic.py)                          | 否     |
| **10 · 工具执行与审批**                 | 工具风险分级；高风险动作前展示真实参数并审批；批准绑定参数；保留执行审计记录                    | [p10a_risk_tiers_approval.py](p10a_risk_tiers_approval.py)                            | 是     |
| **11 · MCP**                     | Server 暴露工具与资源；宿主启动并发现能力；把工具转给模型，再转发调用和错误结果               | [p11a_mcp_server.py](p11a_mcp_server.py)、[p11b_mcp_client.py](p11b_mcp_client.py)     | 客户端需要 |
| **13 · LangChain 家族框架**          | 从链、结构化输出和预置 Agent，到显式图、并行、持久化、记忆和 Deep Agents；详见下一节       | `p13a`～`p13j`，见下表                                                                     | 部分需要  |
| **16 · Context Engineering**     | 给上下文分配预算；压缩工具返回、折叠旧结果、裁剪历史；比较提示布局对可复用前缀的影响                | [p16a_context_budget.py](p16a_context_budget.py)                                      | 否     |
| **16 · 历史压缩** | LangGraph 压缩节点与 RemoveMessage；保护目标、约束、已确认事实、待办及来源，保留完整工具轮次 | [p16b_compaction.py](p16b_compaction.py) | 是 |
| **17 · Agent 记忆**                | 写入、检索、更新、过期与删除；记录来源；避免将外部资料中的指令写成用户记忆                     | [p17a_memory_lifecycle.py](p17a_memory_lifecycle.py)                                  | 否     |
| **18 · 混合检索** | 中文 BM25 + 真实 Embedding + Chroma/Qdrant + RRF；LangChain Retriever、权限过滤与相关性评测 | [p18a_hybrid_retrieval.py](p18a_hybrid_retrieval.py) | 本地 Embedding |
| **18 · 自主检索** | LangGraph 自主查找、改词与停止；查询去重、次数限制、来源白名单 | [p18b_agentic_search.py](p18b_agentic_search.py) | 是 + 本地 Embedding |
| **19 · Mini Harness** | 一个图循环接上校验、审批、工具执行、重复失败检测和上下文 / 调用预算 | [p19_mini_harness.py](p19_mini_harness.py) | 是 |
| **22 · 多 Agent**                 | 把子 Agent 包装成工具；隔离中间上下文；比较主上下文大小、调用次数与累计 token             | [p22a_agent_as_tool.py](p22a_agent_as_tool.py)                                        | 是     |
| **24 · 循环保护与兜底**                 | 步数上限、规范化参数后的重复失败硬停止、累计输入 token 预算；打印实际停止原因                   | [p24a_loop_guard.py](p24a_loop_guard.py)                                              | 是     |
| **25 · Agent 评估**                | 六个任务各跑三次；检查工具结果与回答；区分通过、失败、待复核，以及 pass@3 / pass^3 | [p25a_eval_pass_k.py](p25a_eval_pass_k.py)                                            | 是     |
| **26 · 追踪与观测** | Langfuse CallbackHandler 追踪 LangGraph、模型 generation 与工具 span；用量、故障、脱敏与读回 | [p26_langfuse_tracing.py](p26_langfuse_tracing.py) | 是 + Langfuse |
| **27 · 安全与 Guardrails**          | 间接提示注入如何藏在检索资料中；在工具执行层限制“不可信内容 + 私密数据 + 对外发送”的组合          | [p27a_indirect_injection.py](p27a_indirect_injection.py)                              | 是     |
| **28 · 模型路由与 LLM 网关**            | 按错误类型决定是否重试；指数退避、备用模型、熔断与恢复；使用模拟服务和虚拟时钟                   | [p28a_retry_fallback_breaker.py](p28a_retry_fallback_breaker.py)                      | 否     |
| **29 · 性能与成本优化**                 | 独立工具串行与并发的耗时差异；每个工具单独超时；有依赖的调用必须分轮执行                      | [p29_parallel_tools.py](p29_parallel_tools.py)                                        | 否     |

### 第 13 章：LangChain / LangGraph / Deep Agents

完整的机制讲解与练习见 [LangChain / LangGraph / Deep Agents 教程](LangChain-LangGraph-DeepAgents教程.md)。章节映射对照[现行第 13 篇](https://my.feishu.cn/wiki/ByAnwYnPdiDXq4kBjybcpyWFntd)，核对版本为 2026-10-28 revision 98；扩展机制单独标注。

建议按下表顺序学习，而不是严格按文件名字母排序：

| 顺序 | 小节                  | 讲什么、重点看什么                                                                    | 代码                                                             | 模型 |
|----|---------------------|------------------------------------------------------------------------------|----------------------------------------------------------------|----|
| 1  | LangChain 链式调用      | `Prompt → Model → Parser`；`invoke` 单次、`batch` 批量、`stream` 流式调用               | [p13b_langchain_chain.py](p13b_langchain_chain.py)             | 是  |
| 2  | LangChain 结构化输出     | 用 Pydantic 接收模型抽取结果；区分字段合法与订单归属合法                                            | [p13c_langchain_structured.py](p13c_langchain_structured.py)   | 是  |
| 3  | LangChain Agent     | `create_agent` 管理工具循环；观察工具请求和返回；同一 `thread_id` 延续会话                          | [p13d_langchain_agent.py](p13d_langchain_agent.py)             | 是  |
| 4  | LangGraph 基础        | State、节点、条件边、reducer、内存检查点；审批中断和节点重放                                         | [p13a_langgraph_basics.py](p13a_langgraph_basics.py)           | 否  |
| 5  | 显式 ReAct 图          | `MessagesState`、`bind_tools`、`ToolNode`；模型与工具间的循环边；`updates` 和 `messages` 事件 | [p13e_langgraph_react.py](p13e_langgraph_react.py)             | 是  |
| 6  | 动态并行与汇总             | 先观察并发写冲突，再通过 reducer、`Send` 和 deferred 节点完成 map-reduce                       | [p13f_langgraph_map_reduce.py](p13f_langgraph_map_reduce.py)   | 否  |
| 7  | 持久化与审批恢复            | SQLite 保存检查点；两个独立进程完成中断和恢复；`Command` 路由；审批节点从头重跑                             | [p13g_langgraph_persistence.py](p13g_langgraph_persistence.py) | 否  |
| 8  | 会话状态与长期记忆           | Checkpointer 管会话进度，Store 管跨会话数据；Runtime Context 注入身份；用户隔离                    | [p13h_langgraph_store.py](p13h_langgraph_store.py)             | 否  |
| 9  | 时间旅行                | 找到历史检查点；用 `update_state` 创建新分支；使用返回的新配置继续，保留原始结果                             | [p13j_langgraph_time_travel.py](p13j_langgraph_time_travel.py) | 否  |
| 10 | Deep Agents Harness | 主 Agent 委派子任务；虚拟文件、Skills、Memory、可选任务规划；工具写入前暂停审批                            | [p13i_deepagents_harness.py](p13i_deepagents_harness.py)       | 是  |

三层接口的分工：**LangChain 提供模型组件与常用 Agent，LangGraph 提供显式状态和流程控制，Deep Agents 提供长任务所需的工作区与配套能力。**

## 5. 常用运行方式

### 单独运行一课

```bash
uv run p07a_react_loop.py
uv run p13b_langchain_chain.py
```

### 运行框架专题

```bash
# 无需模型：p13a、p13f、p13g、p13h、p13j
bash run_frameworks.sh --offline

# 完整专题：先配置 .env，会调用真实模型
bash run_frameworks.sh --all
```

### 运行全部示例

```bash
bash run_all.sh
```

此脚本按文件名运行基础 `p*.py`，跳过由客户端启动的 MCP Server，默认也跳过需要下载 Embedding 的 p18a / p18b 和会上传追踪的 p26。任一示例失败便停止，日志保存到 `/tmp/run_all_文件名.log`。它是批量演示入口，不是固定模型效果的测试断言。

### 代码检查

```bash
uv run ruff check .
```

ruff 由 `uv sync` 一并装进 `.venv`，规则见 [ruff.toml](ruff.toml)。

可选集成和 B 批本地检查：

```bash
# 7 个 --check：不调用模型，不下载 Embedding，不上传 Langfuse
bash run_all.sh --check-b

# 显式包含向量检索 / Langfuse 集成，需要相应模型和服务配置
bash run_all.sh --with-retrieval --with-tracing
```

### 几个需要特别说明的入口

**MCP：运行客户端即可，客户端会启动 Server。**

```bash
uv run p11b_mcp_client.py
```

`p11a_mcp_server.py` 使用 stdio 通信，单独启动会等待客户端输入，不会像普通示例一样直接打印答案。

**第 10 章：单独在终端运行时交互审批。**

```bash
uv run p10a_risk_tiers_approval.py
```

`--approve` 参数用于教学时自动批准；`run_all.sh` 关闭标准输入，未显式批准的高风险操作按脚本的非交互规则拒绝。

**LangGraph：手动体验关闭进程后的恢复。**

```bash
uv run p13g_langgraph_persistence.py start --db /tmp/shop-approval.sqlite --thread-id lesson-1
uv run p13g_langgraph_persistence.py approve --db /tmp/shop-approval.sqlite --thread-id lesson-1
```

第二条命令中的 `approve` 可以替换为 `reject`。同一数据库和 thread 的任务只能启动一次，再次练习使用新的 `--thread-id`。不传参数时，脚本自动用临时数据库完成两个进程的演示。

**Deep Agents：观察拒绝文件写入后的行为。**

```bash
uv run p13i_deepagents_harness.py --decision reject
```

默认批准与 `--decision reject` 都是预设的教学决定，不是真实人工审核。`/reports/order.md` 保存在 StateBackend 的虚拟文件状态里，不是电脑上的同名路径。

## 6. 目录与公共模块

| 文件                                     | 职责                                         |
|----------------------------------------|--------------------------------------------|
| [pyproject.toml](pyproject.toml)       | Python 版本要求与全部依赖，`uv sync` 按它配置环境            |
| [uv.lock](uv.lock)                     | 锁定的依赖版本，由 uv 生成，随代码提交，不要手改                  |
| [.python-version](.python-version)     | 本项目使用的 Python 版本（3.12）                        |
| [setting.py](setting.py)               | 配置字段、`.env` 读取、类型校验和共享 `settings` 实例       |
| [.env.example](.env.example)           | 不含真实密钥的配置模板                                |
| `_llm.py`                              | OpenAI 兼容客户端、消息与工具调用结构、最小 Agent 循环         |
| `_langchain.py`                        | 创建 `ChatOpenAI`，将本地订单和物流能力包装成 LangChain 工具 |
| `_shop.py`                             | 小店演示数据、工具 Schema、参数校验、当前顾客的订单归属检查          |
| `p*.py`                                | 各章节可独立运行的教学示例                              |
| [run_frameworks.sh](run_frameworks.sh) | 按学习顺序运行第 13 章，支持无模型与完整模式                   |
| [run_all.sh](run_all.sh)               | 按文件名运行基础示例，可选检索 / 追踪集成或 B 批本地检查，失败保留日志                     |
| `llm_env.sh`                           | 旧版项目外配置启动器，保留兼容；新流程直接使用 `.env`             |
| [ruff.toml](ruff.toml)                 | Python 代码检查规则                              |

本目录用于理解单个机制；旁边的 [`专栏代码/`](../专栏代码/) 用于练习文章级项目与工程写法，[`shop-assistant/`](../shop-assistant/) 用于构建完整客服产品。

## 7. 如何观察结果与排查问题

模型生成内容、工具调用次数、评估通过率可能随模型和每次运行变化。循环保护不保证每次触发，提示注入也不保证模型每次上当；重点看实际工具轨迹和代码中的控制条件。

订单、物流、退款和发信使用本地演示数据或模拟动作，不会实际退款或发信。`p28a` 的服务错误和等待时间也是模拟的。内存 Checkpointer 和 Store 在进程退出后消失；跨进程恢复请观察 SQLite 示例。

| 遇到的问题                            | 检查方法                                                |
|----------------------------------|-----------------------------------------------------|
| `uv: command not found`          | 确认 uv 已安装，安装目录已加入 PATH，必要时重开终端                      |
| 找不到 `.py` 文件                     | 确认当前目录为本 README 所在的 `agent-learning`                |
| `ModuleNotFoundError`            | 先运行 `uv sync`，再用 `uv run 文件名.py`；用 IDE 运行时检查解释器是否为 `.venv` 里的 Python |
| 提示缺少 `LLM_API_KEY` 或 `LLM_MODEL` | 检查 `.env` 是否存在、字段是否非空、是否仍是模板占位值                     |
| 修改 `.env` 后仍使用旧值                 | 重启进程；检查是否有已有环境变量或旧 `llm_env.sh` 覆盖配置                |
| 新增的配置读不到                         | 先在 `Settings` 中声明字段；仅添加 `.env` 键不会自动生成属性            |
| 配置类型校验失败                         | 检查布尔、数字等字段是否填写合法值，不要随意关闭校验                          |
| API 返回 401 / 403                 | 核对密钥、模型权限和供应商配置                                     |
| API 返回 404                       | 检查 base URL、是否需要 `/v1`，以及模型名称                       |
| 不支持工具调用或流式输出                     | 确认所选模型和供应商接口具备该示例要求的能力                              |
| `p13f` 打印 `InvalidUpdateError`   | 第一段故意触发并捕获该异常，用来说明并行写入为什么需要 reducer                 |
| 找不到待审批任务                         | 检查数据库路径、thread_id 和任务是否已处理                          |
| `GraphRecursionError`            | 先查看循环退出条件和工具错误，避免只调大步数上限                            |
| 批量运行失败                           | 查看脚本提示的 `/tmp/run_all_*.log`，再单独运行失败文件              |

## 8. 官方学习文档

LangChain
```markdown
https://docs.langchain.com/oss/python/langchain/overview
```
LangGraph
```markdown
https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
```
DeepAgents
```markdown
https://docs.langchain.com/oss/python/deepagents/overview
```

代码检查：
```bash
uvx ruff check .
bash -n run_frameworks.sh run_all.sh
```

写新示例时，继续使用“一个文件讲一个机制”的约定：文件名对应章节、开头说明运行方式和观察点、末尾提供可修改的练习。

## 9. 新增机制示例（B 批 7 个已实现，C / D 批 24 个待实现）

保留真实模型调用，不增加剧本模式。A 批修改现有示例；B 批已新增下列 7 个文件；C / D 批仍是计划，待逐批验收。每个示例包含运行命令、观察点和代码练习。

### B 批：结合 LangChain / LangGraph 的工程机制（7 个已实现）

| 文件 | 实现与验收重点 |
|---|---|
| [p03a_prompt_layers.py](p03a_prompt_layers.py) | system / 业务约束 / 检索资料 / 用户问题分层；严格模板缺变量报错；外部资料不提升为指令 |
| [p05b_streaming.py](p05b_streaming.py) | 真实文本流、首事件与首文本延迟、用量；按 index 合并交错 tool_calls 分片，完整解析后执行并回填 |
| [p16b_compaction.py](p16b_compaction.py) | 折叠旧过程，保留目标、约束、已确认事实、待办、来源；保持调用与工具结果配对，验证压缩后能继续完成任务 |
| [p18a_hybrid_retrieval.py](p18a_hybrid_retrieval.py) | 中文分词 BM25 + Chroma/Qdrant 向量检索 + RRF；同一文档集、Embedding 和问题集，输出三路排名、来源、Recall@k 与 MRR |
| [p18b_agentic_search.py](p18b_agentic_search.py) | 模型自主决定关键词、重查与停止；使用 p18a 的检索接口，限制轮数与去重，按工具轨迹统计检索次数和引用 |
| [p19_mini_harness.py](p19_mini_harness.py) | 单一循环按钩子组合参数校验、审批、重复失败保护、上下文与调用预算；审批绑定真实参数，拒绝与预算耗尽均有明确结果 |
| [p26_langfuse_tracing.py](p26_langfuse_tracing.py) | Langfuse 根任务 trace、模型 generation、工具 span、用量、耗时与错误；结束 flush，展示如何定位一次失败，不上传明文密钥与个人信息 |

先读 `p13b` / `p13e` 的链与图基础，再读本批机制：p03a 用 `ChatPromptTemplate → ChatOpenAI → Parser`，p05b 用 `AIMessageChunk` 合并流；p16b / p18b / p19 用显式 StateGraph 节点和条件边表达控制规则；p26 的 CallbackHandler 记录这些框架调用。

```bash
uv run p03a_prompt_layers.py --render-only
uv run p03a_prompt_layers.py
uv run p05b_streaming.py
# 供应商不支持流式 usage 时：增加 --without-usage
uv run p16b_compaction.py --budget 2200 --summary-budget 8000
uv run p18a_hybrid_retrieval.py --backend both
uv run p18b_agentic_search.py --backend qdrant
uv run p19_mini_harness.py --decision reject
uv run p26_langfuse_tracing.py --upload
```

**历史压缩：** `protected` 单独存储目标、约束、已确认事实和待办，摘要节点不允许改写；摘要按低可信资料注入，不提升为 system 指令。只折叠已完成旧轮次，保留最近两轮，校验 tool call / result 配对。摘要调用本身也有独立预算；压缩失败就停止，不靠丢证据继续。字节估计不是精确 tokenizer 或计费数据。

**检索配置与对照：** 正常运行使用 FastEmbed 的 `BAAI/bge-small-zh-v1.5`，首次约下载 100 MB；不需要聊天供应商提供 Embedding 接口。可在现有 `.env` 设置 `EMBEDDING_MODEL` 为 FastEmbed 支持的其他模型。Chroma 用临时本地集合，Qdrant 默认 `:memory:`；可选 `QDRANT_URL` / `QDRANT_API_KEY` 接远程实例，每次只清理本例自建的随机集合。两后端复用同一模型、文档集、问题集与向量。

中文 jieba 分词后做 BM25；BM25 和向量后端都先按应用固定租户及可选分类过滤，再各取候选，以 `1 / (60 + rank)` 做 RRF。终端显示三路排名、分数、来源和人工标签的 Recall@3 / MRR@3。向量近邻不等于相关，RRF 也不保证每题更好；小数据集用于解释机制。p18b 把 Retriever 包成工具，让模型自主决定查词、重查与停止，并限制实际查询次数、缓存重复查询、校验引用来自已返回资料。引用存在不等于结论已被证明。

p18a / p18b 使用独立接口，与 `_shop` 的共同汉字简易检索对照，不给其他入门文件增加下载模型的要求。`--check` 使用固定数值向量验证真实数据库的过滤与排序接口，**不测试语义效果，也不作为正常运行的 Embedding 替代品**。

**Harness 审批：** p19 的 `interrupt` 展示本次工具名、规范化参数与指纹，恢复时必须匹配；审批之前不执行动作。整批参数验证完才开始执行，重复失败两次则跳过剩余调用；上下文、累计输入、模型轮数、工具次数都有上限。退款工具仅在可检查点化的状态里创建本地草稿，不会实际退款或提交。默认终端手动确认，非交互拒绝；`--decision approve` 仅用于教学预置决定。真实外部写入还需服务端幂等键与审批身份认证。

**Langfuse 配置：** p26 固定 SDK 3.x，使用支持该版本的服务端；不要混用其他 SDK 大版本的 API。沿用现有 `.env`，加入以下字段，不需要额外 shell 配置：

```dotenv
LANGFUSE_BASE_URL=https://你的Langfuse实例地址
LANGFUSE_PUBLIC_KEY=你的项目PublicKey
LANGFUSE_SECRET_KEY=你的项目SecretKey
```

`RetrievalSettings` / `TracingSettings` 是示例内的 `Settings` 子类，只增加该示例的字段，复用现有 `.env` 加载规则。p26 必须加 `--upload` 才发送固定的公开教学问答；不使用订单、手机号或地址工具。可观察根任务、图节点、模型 generation、工具 span、用量、耗时和错误；正则脱敏仅作补充，不是完整隐私保障。调用 `flush` 后尝试按 trace_id 读回，区分“已请求刷新”与“平台已可查询”；费用数据仍依赖平台支持所用模型的价格配置。

### C 批：补全篇目中的机制（11 个）

| 文件 | 实现与验收重点 |
|---|---|
| `p07b_reflection.py` | 初稿→反思→修订；反馈必须指向事实或规则，限制次数，对比修订前后结果 |
| `p08b_parallel_workers.py` | 按子问题并行调用独立 worker；隔离上下文、限并发、处理单路超时并稳定汇总 |
| `p08c_evaluator_optimizer.py` | 独立评估器按明确 rubric 返回结构化反馈；优化器按反馈重写，满足标准或达到上限停止 |
| `p10b_subprocess_limits.py` | 子进程墙钟超时、CPU / 内存限制与平台差异；清理进程组，演示它无法限制的文件及网络访问 |
| `p12_skill_loader.py` | 安全解析 YAML frontmatter，先展示名称与描述，选中后加载正文；校验路径、缺字段和畸形元数据 |
| `p22b_handoff.py` | 在客服与售后 Agent 间转交执行权；白名单状态和身份由程序传递，限制往返次数 |
| `p22c_generator_verifier.py` | 生成器给答案与证据；验证器独立查工具验证事实，失败时退回或结束；区别于文字润色 |
| `p24b_structured_errors.py` | 统一错误码、可重试性、用户可见说明、内部诊断；分开处理参数、业务、服务与策略错误 |
| `p25b_llm_judge_agreement.py` | 固定带人工标签的样本与 rubric；结构化判分，计算一致率和混淆矩阵，展示分歧而非把 judge 当真值 |
| `p27b_output_guard_pii.py` | 输出前检查手机号、邮箱等信息并按策略遮盖；覆盖误报与漏报，流式输出按缓冲边界检测，避免先泄露后拦截 |
| `p28b_hybrid_router.py` | 代码规则筛选能力与风险，模型处理模糊意图；限制候选、记录理由、处理非法输出，对比固定路由 |

`p19` 先用已有错误格式完成组装；`p24b` 实现后再接入其统一错误结构。`p08b` 演示任务分解与并行，`p22b` 演示执行权转移，避免仅换名称重复同一个工具循环。

### D 批：概念、模型差异、框架与扩展（13 个）

| 文件 | 实现与验收重点 |
|---|---|
| `p01_chat_vs_agent.py` | 同一问题比较纯对话与工具循环，展示谁获得了外部事实、谁执行动作 |
| `p02_workflow_vs_agent.py` | 同一任务比较固定流程与模型自主选路，用执行轨迹说明控制权 |
| `p03b_few_shot_cot.py` | 无示例 / few-shot / 分步解题指令对照；评估答案和可核验依据，不声称拿到模型内部思维链 |
| `p04a_three_ways_json.py` | 提示约束、JSON mode、JSON Schema 对照；分别检查可解析性、Schema 和业务规则，显式报告不支持的能力 |
| `p05c_prompt_cache_hit.py` | 重复稳定前缀、改变末尾输入，读取供应商实际 cached_tokens；未返回指标时标为不可观测，命中不作保证 |
| `p06_reasoning_effort.py` | 对支持的模型比较不同 effort 的正确率、延迟与用量；不支持则明确跳过，不虚构推理 token |
| `p09b_tool_desc_ab.py` | 固定工具 Schema 和评测问题，只变更说明；多次采样统计工具选择与参数准确率 |
| `p13k_langchain_middleware.py` | wrap_model_call / wrap_tool_call 等真实钩子，演示模型选择、工具限制和结构化错误，对照手写 harness |
| `p14_openai_agents_sdk.py` | 真实 Runner、function_tool、会话与 guardrail；模型能力和 trace 上传分别配置，避免与 Langfuse 示例混淆 |
| `p23a_a2a_card_task.py` | 本地启动服务与客户端，读取 Agent Card，创建并查询 Task，观察状态变化与产物；注明协议版本 |
| `p23b_ag_ui_events.py` | 本地事件流与消费者，发送 RUN / TEXT_MESSAGE / TOOL_CALL / STATE 等事件；检查 id、顺序、结束与错误事件 |
| `p30_playwright_browser_agent.py` | 最后实现：本地演示网页，真实浏览器观察与操作循环，域名及动作限制；浏览器下载为显式准备步骤 |
| `p31_image_input.py` | 最后实现：本地图片输入到支持视觉的模型，处理图像格式、大小和能力不支持；对比提取结果与可核验内容 |

01、02 虽在 D 批开发，最终学习顺序仍放在最前。p30 / p31 单独作为可选扩展，基础批量入口不自动安装浏览器或调用视觉模型；浏览器下载量随平台与版本变化。

### 每批验收方式

- 静态检查、`pyproject.toml` 依赖与运行说明一致；默认入口不伪造模型返回值。
- 真实模型示例记录所用模型与实际现象，不把模型未触发某条路径说成防线已通过验证。
- 参数、审批、故障停止等规则用本地回归检查验证；模型效果实验保留原始回答、工具轨迹及待复核结果。
- 新依赖、模型下载和外部服务在对应示例说明；基础批量运行与可选扩展分开。

A 批新增的本地规则检查入口（均不调用模型）：

```bash
uv run p24a_loop_guard.py --check-guards
uv run p25a_eval_pass_k.py --check-graders
```

A 批验收（2026-10-28）：Ruff、Python 语法检查、Shell 语法检查、五个无模型框架示例通过；
评分器检查覆盖 24 个回答案例与 6 组聚合结果，重复失败计数覆盖 8 项，另验证了循环执行停止的 6 条路径。
p24 / p25 / p27 的真实模型运行在客户端初始化时因缺少有效模型配置而停止，本次未验证模型效果；配置文件未修改。


B 批验收（2026-10-28）：七个示例均通过独立 PEP 723 环境下的 `--check` 与 `--help`；Ruff、Python 语法、Shell 语法及 diff 空白检查通过。LangGraph 本地检查覆盖真实压缩节点、工具消息配对、检索去重和预算、审批暂停/恢复与参数绑定、整批参数校验、重复失败硬停止；Langfuse SDK 实际生成根任务、generation、工具及错误 span，仅导出到内存。

已实际下载并运行 `BAAI/bge-small-zh-v1.5`，Chroma / Qdrant 本地模式使用相同向量完成 5 题评测，两者均得到：BM25 Recall@3=0.80 / MRR@3=0.70，向量 1.00 / 1.00，RRF 1.00 / 0.90。第一题“买回来坏了，寄回去谁付钱”中，BM25 漏召回质量政策；RRF 补回了正确来源，却把它排在向量结果之后，保留此例用于分析排序退化。该结果只代表本例固定小样本。

当前环境缺少完整的模型与 Langfuse 凭据，尚未验证真实聊天模型生成及 Langfuse 服务端入库；本地检查结果不代表这些在线集成已成功。现有 `.env` 加载与公共模型配置未修改。本批停在 B，不包含 C / D 实现。
