"""
思维链推理示例 - Reasoning Chain of Thought

演示如何使用 LangChain 构建思维链提示词，引导模型在回答前进行预调查分析。
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import httpx

# 思维链提示词模板：引导模型将事实分为四类进行分析
FACTS_PROMPT = """下面我将向您提出一个请求。在开始处理该请求之前，请先回答以下预调查问题，尽您所能作答。请记住，您在 trivia（常识问答）方面具有 Ken Jennings 级别的水平，在谜题方面具有 Mensa（门萨俱乐部）级别的水平，因此应该有丰富的知识储备可供挖掘。

以下是该请求：

{task}

以下是预调查问题：

    1. 请列出请求中明确给出的任何具体事实或数据。可能没有任何此类信息。
    2. 请列出可能需要查阅的任何事实，以及具体可以在哪里找到这些信息。在某些情况下，请求本身会提及权威来源。
    3. 请列出可能需要推导的任何事实（例如，通过逻辑演绎、模拟或计算得出）。
    4. 请列出从记忆中回忆出的任何事实、直觉、经过充分推理的猜测等。

在回答此调查时，请记住，"事实"通常是具体的名称、日期、统计数据等。您的回答应使用以下标题：

    1. 已给出或已验证的事实
    2. 需要查阅的事实
    3. 需要推导的事实
    4. 有根据的猜测

不要在您的回复中包含其他任何标题或部分。在被要求之前，不要列出下一步行动或计划。
"""

# 示例任务
task = "我要去北京玩三天"

# 使用 LangChain PromptTemplate 生成完整提示词
facts_template = PromptTemplate.from_template(FACTS_PROMPT)
full_prompt = facts_template.format(task=task.strip())

# 配置并调用大语言模型
llm = ChatOpenAI(
    model="gpt-oss-120b",
    temperature=0,  # 降低随机性以提升逻辑稳定性
    base_url="你的url",
    api_key="你的api_key",
    http_client=httpx.Client(verify=False)  # 仅用于测试环境
)

# 发送请求并获取响应
response = llm.invoke([HumanMessage(content=full_prompt)])

# 输出模型的思维链推理结果
print(response.content)
