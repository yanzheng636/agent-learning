# /// script
# requires-python = ">=3.12"
# dependencies = ["langchain-openai>=1.7,<2", "python-dotenv>=1.1,<2", "pydantic-settings>=2.10,<3"]
# ///
"""资料库 13 · LangChain 家族配套 · LangChain 入门：Prompt → Model → Parser。

uv run p13b_langchain_chain.py

看什么：同一条 LCEL 链的 invoke、batch、stream；流程由代码决定，不是 Agent。
需要真实模型配置；本例共发起四次模型请求。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import _shop
from _langchain import make_model


def main() -> None:
    # ---- 1. 提示模板：固定角色 + 每次传入的问题 ----
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是小店客服。只依据以下政策回答，不知道就说不知道。政策：{policy}"),
        ("human", "{question}"),
    ]).partial(policy=_shop.KNOWLEDGE[0]["text"])
    chain = prompt | make_model() | StrOutputParser()
    # dict → ChatPromptValue → AIMessage → str；| 表示把上一步输出传给下一步。

    print("==== 1. invoke：一次输入得到一个字符串 ====")
    answer = chain.invoke({"question": "质量问题退货，运费谁出？"})
    print(type(answer).__name__, answer)

    print("\n==== 2. batch：独立问题并发，结果按输入顺序返回 ====")
    questions = ["无理由退货的期限是多久？", "无理由退货运费谁出？"]
    answers = chain.batch(
        [{"question": q} for q in questions], config={"max_concurrency": 2},
    )
    for question, answer in zip(questions, answers, strict=True):
        print(f"  {question} → {answer}")

    print("\n==== 3. stream：边收到文本片段边打印 ====")
    for chunk in chain.stream({"question": "用两句话总结退货规则。"}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()

# 试试改：
# - 去掉 StrOutputParser()，观察 invoke 返回的 AIMessage 的 content 和 usage_metadata。
# - 在模型之前接一个 RunnableLambda 做输入检查，和 p08a 的代码关卡对照。
