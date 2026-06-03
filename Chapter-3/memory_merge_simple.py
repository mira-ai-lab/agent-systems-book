"""
记忆合并示例 - Memory Merge（语义融合）

演示如何使用 LangChain 引导模型将多条相似记忆合并为一条精炼的总结。
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import httpx
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ---------------------------------------------------------------------------
# 1. 思维链提示词模板：引导模型处理冲突并去重
# ---------------------------------------------------------------------------
MEMORY_MERGE_PROMPT = """请将以下多条关于同一主题的用户记忆合并为一条综合记忆。

原始记忆片段：
{memories}

请像整理课堂笔记一样思考：
1. 识别重复信息并进行去重。
2. 如果存在冲突（如预算从800变为1000），以最新的信息为准。
3. 保持语言简洁、连贯，保留所有关键的硬性约束和偏好。

请在文末单独给出且仅给出一个 JSON 对象（不要 markdown 代码块），字段如下：
{{
  "summary": "合并后的一句话总结",
  "key_points": ["原子要点1", "原子要点2"],
  "importance": 0.0到1.0之间的数字,
  "reason": "一句话说明为什么这样合并"
}}
"""

# ---------------------------------------------------------------------------
# 2. 示例数据：两条高度相似的住宿偏好
# ---------------------------------------------------------------------------
old_memory = "用户喜欢住海景房，预算每晚不超过800元。"
new_memory = "用户再次强调：一定要住海景酒店，每晚800块以内，必须有WiFi。"

memories_text = f"1. {old_memory}\n2. {new_memory}"

# ---------------------------------------------------------------------------
# 3. 配置并调用大语言模型
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
    temperature=0,
    base_url=os.getenv("DASHSCOPE_CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    http_client=httpx.Client(verify=False)
)

# 生成完整提示词
merge_template = PromptTemplate.from_template(MEMORY_MERGE_PROMPT)
full_prompt = merge_template.format(memories=memories_text)

print("=" * 70)
print("【发给大模型的完整提示词】")
print("-" * 70)
print(full_prompt)
print("\n" + "-" * 70)

# 发送请求并获取响应
print("【模型输出：合并后的记忆 + JSON】\n")
response = llm.invoke([HumanMessage(content=full_prompt)])
print(response.content)

print("\n" + "=" * 70)
print("演示结束。在实际工程中，我们会用这个 JSON 更新 Chroma 中的原记录。")
# ======================================================================
# 【发给大模型的完整提示词】
# ----------------------------------------------------------------------
# 请将以下多条关于同一主题的用户记忆合并为一条综合记忆。

# 原始记忆片段：
# 1. 用户喜欢住海景房，预算每晚不超过800元。
# 2. 用户再次强调：一定要住海景酒店，每晚800块以内，必须有WiFi。

# 请像整理课堂笔记一样思考：
# 1. 识别重复信息并进行去重。
# 2. 如果存在冲突（如预算从800变为1000），以最新的信息为准。
# 3. 保持语言简洁、连贯，保留所有关键的硬性约束和偏好。

# 请在文末单独给出且仅给出一个 JSON 对象（不要 markdown 代码块），字段如下：
# {
#   "summary": "合并后的一句话总结",
#   "key_points": ["原子要点1", "原子要点2"],
#   "importance": 0.0到1.0之间的数字,
#   "reason": "一句话说明为什么这样合并"
# }


# ----------------------------------------------------------------------
# 【模型输出：合并后的记忆 + JSON】

# {
#   "summary": "用户偏好海景房，预算每晚不超过800元，且必须配备WiFi。",
#   "key_points": ["海景房", "每晚≤800元", "必须有WiFi"],
#   "importance": 0.95,
#   "reason": "两条记忆均一致强调海景、800元预算上限和WiFi为刚性要求，无冲突或更新，仅需去重整合，所有要素均为明确硬性约束。"
# }

# ======================================================================
# 演示结束。在实际工程中，我们会用这个 JSON 更新 Chroma 中的原记录。
