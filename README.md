# AI Agent 技术资料库 · 示例代码

这是一套面向 Python 开发者的 Agent 机制教程，配合飞书「AI Agent 技术资料库」学习。业务场景统一为**小店客服助手**：查订单、查物流、回答政策、处理售后和生成报告。

每个示例围绕一个机制展开，包含中文说明、运行输出和“试试改”练习。建议先运行观察，再阅读代码，最后修改参数对比结果。

> 文件名中的 `p04`、`p13` 对应资料库章节编号。本目录只包含已经实现的配套示例，编号不连续；下方按现有代码说明各章内容。

## 1. 环境准备

### 1.1 安装 uv 和 Python

需要 **Python 3.12+**。推荐使用 `uv` 管理 Python 和脚本依赖；批量运行脚本需要 Bash，Windows 用户可在 WSL 中运行。

macOS 使用 Homebrew 安装 uv：

```bash
brew install uv
```

Windows
也可以在已有 Python 的环境中安装：

```bash
python3 -m pip install --user uv
```

安装后确认 `uv` 在 PATH 中，并准备 Python：

```bash
uv --version
uv python install 3.12
```

**下文命令都在本 README 所在的 `agent-learning` 目录执行。** 如果终端当前位于本仓库根目录：

脚本开头的 `# /// script` 区块声明了 Python 版本和依赖。使用 `uv run 文件名.py` 时，uv 会自动准备环境，不需要手动建虚拟环境或统一安装所有包。

### 1.2 先验证无模型示例

还没有 API Key，也可以先运行：

```bash
uv run p13a_langgraph_basics.py
```

这个示例用普通 Python 函数演示状态、分支和审批。也可以一次运行五个无模型的框架示例：

```bash
bash run_frameworks.sh --offline
```

`--offline` 指**不调用模型**，不等于完全断网：首次运行仍需下载依赖；使用 tiktoken 的其他示例首次运行还可能下载编码文件。

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
| 基础机制   | 04 → 05 → 07 → 08 → 09 → 10 → 11       | 理解模型、工具、循环、控制流和 MCP 如何协作   |
| 框架实践   | 13：LangChain → LangGraph → Deep Agents | 将手写机制映射到框架 API，掌握状态和恢复     |
| 上下文与记忆 | 16 → 17 → 22                           | 控制上下文体积，管理记忆，理解子 Agent 的代价 |
| 可靠性与优化 | 24 → 25 → 27 → 28 → 29                 | 防循环、做评估、拦注入、处理故障并降低延迟      |

每章都可以单独运行。建议每次回答三个问题：**输入是什么、由谁决定下一步、执行结果存在哪里？**

## 4. 每一章讲什么

下表“模型”表示是否调用真实模型服务。标为“否”的示例不需要 API Key；标为“是”的示例会产生模型调用费用。

| 章节                               | 主要内容与观察点                                                  | 对应代码                                                                                  | 模型    |
|----------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------|-------|
| **04 · 结构化输出与 Function Calling** | 工具定义如何发给模型；模型只提出调用请求；执行后用 `tool_call_id` 对齐工具结果，再请求最终回答   | [p04b_tool_call_roundtrip.py](p04b_tool_call_roundtrip.py)                            | 是     |
| **05 · Token、成本与调用约束**           | 中英文 token 对比；提示、工具和历史分别占多少；多轮 Agent 的累计输入成本               | [p05a_count_tokens.py](p05a_count_tokens.py)                                          | 否     |
| **07 · Agent 循环**                | ReAct 每步决策与步数上限；Plan-and-Execute 先生成计划，再按依赖分批并发执行，最后汇总    | [p07a_react_loop.py](p07a_react_loop.py)、[p07c_plan_execute.py](p07c_plan_execute.py) | 是     |
| **08 · 常用编排模式**                  | 固定链式流程“抽取 → 业务关卡 → 回复”；按问题类别路由；区分代码控制流程与模型自主选动作           | [p08a_chain_router.py](p08a_chain_router.py)                                          | 是     |
| **09 · 工具设计**                    | Pydantic 生成参数 Schema；处理缺字段、多字段、格式错误；把错误回填给模型；压缩工具结果       | [p09a_tool_schema_pydantic.py](p09a_tool_schema_pydantic.py)                          | 否     |
| **10 · 工具执行与审批**                 | 工具风险分级；高风险动作前展示真实参数并审批；批准绑定参数；保留执行审计记录                    | [p10a_risk_tiers_approval.py](p10a_risk_tiers_approval.py)                            | 是     |
| **11 · MCP**                     | Server 暴露工具与资源；宿主启动并发现能力；把工具转给模型，再转发调用和错误结果               | [p11a_mcp_server.py](p11a_mcp_server.py)、[p11b_mcp_client.py](p11b_mcp_client.py)     | 客户端需要 |
| **13 · LangChain 家族框架**          | 从链、结构化输出和预置 Agent，到显式图、并行、持久化、记忆和 Deep Agents；详见下一节       | `p13a`～`p13j`，见下表                                                                     | 部分需要  |
| **16 · Context Engineering**     | 给上下文分配预算；压缩工具返回、折叠旧结果、裁剪历史；比较提示布局对可复用前缀的影响                | [p16a_context_budget.py](p16a_context_budget.py)                                      | 否     |
| **17 · Agent 记忆**                | 写入、检索、更新、过期与删除；记录来源；避免将外部资料中的指令写成用户记忆                     | [p17a_memory_lifecycle.py](p17a_memory_lifecycle.py)                                  | 否     |
| **22 · 多 Agent**                 | 把子 Agent 包装成工具；隔离中间上下文；比较主上下文大小、调用次数与累计 token             | [p22a_agent_as_tool.py](p22a_agent_as_tool.py)                                        | 是     |
| **24 · 循环保护与兜底**                 | 步数上限、重复失败检测、token 预算三道限制；由执行代码决定何时停止和追问                   | [p24a_loop_guard.py](p24a_loop_guard.py)                                              | 是     |
| **25 · Agent 评估**                | 六个任务各跑三次；评分器检查结果与调用过程；区分“至少一次成功”的 pass@3 和“三次全成功”的 pass^3 | [p25a_eval_pass_k.py](p25a_eval_pass_k.py)                                            | 是     |
| **27 · 安全与 Guardrails**          | 间接提示注入如何藏在检索资料中；在工具执行层限制“不可信内容 + 私密数据 + 对外发送”的组合          | [p27a_indirect_injection.py](p27a_indirect_injection.py)                              | 是     |
| **28 · 模型路由与 LLM 网关**            | 按错误类型决定是否重试；指数退避、备用模型、熔断与恢复；使用模拟服务和虚拟时钟                   | [p28a_retry_fallback_breaker.py](p28a_retry_fallback_breaker.py)                      | 否     |
| **29 · 性能与成本优化**                 | 独立工具串行与并发的耗时差异；每个工具单独超时；有依赖的调用必须分轮执行                      | [p29_parallel_tools.py](p29_parallel_tools.py)                                        | 否     |

### 第 13 章：LangChain / LangGraph / Deep Agents

完整的机制讲解与练习见 [LangChain / LangGraph / Deep Agents 教程](LangChain-LangGraph-DeepAgents教程.md)。LangGraph 和 Deep Agents
内容根据[飞书原文](https://my.feishu.cn/wiki/TjZowqX2piL1LikMe0zc4781ncd)改写为独立示例。

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

此脚本按文件名运行所有 `p*.py`，跳过由客户端启动的 MCP Server。任一示例失败便停止，日志保存到 `/tmp/run_all_文件名.log`。它是批量演示入口，不是固定模型效果的测试断言。

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
| [setting.py](setting.py)               | 配置字段、`.env` 读取、类型校验和共享 `settings` 实例       |
| [.env.example](.env.example)           | 不含真实密钥的配置模板                                |
| `_llm.py`                              | OpenAI 兼容客户端、消息与工具调用结构、最小 Agent 循环         |
| `_langchain.py`                        | 创建 `ChatOpenAI`，将本地订单和物流能力包装成 LangChain 工具 |
| `_shop.py`                             | 小店演示数据、工具 Schema、参数校验、当前顾客的订单归属检查          |
| `p*.py`                                | 各章节可独立运行的教学示例                              |
| [run_frameworks.sh](run_frameworks.sh) | 按学习顺序运行第 13 章，支持无模型与完整模式                   |
| [run_all.sh](run_all.sh)               | 按文件名批量运行所有示例，失败停止并保留日志                     |
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
| `ModuleNotFoundError`            | 使用 `uv run 文件名.py`，普通 `python 文件名.py` 不会自动安装脚本声明的依赖 |
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

代码检查：

```bash
uvx ruff check .
bash -n run_frameworks.sh run_all.sh
```

写新示例时，继续使用“一个文件讲一个机制”的约定：文件名对应章节、开头说明运行方式和观察点、末尾提供可修改的练习。
