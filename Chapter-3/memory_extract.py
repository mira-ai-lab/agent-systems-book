"""
记忆抽取示例 - Memory Extraction（原子事实抽取）

只演示如何从对话中抽取可入库的结构化记忆，
不连接 Chroma、不做合并、存储、检索。

范式说明：
  - 传统抽取：原句照抄入库
  - 本示例：对话 → 预分析（四类标题）→ JSON（summary / key_points / importance）

依赖：
    pip install python-dotenv langchain-openai langchain-core httpx

.env：
    DASHSCOPE_API_KEY=你的百炼密钥
    DASHSCOPE_CHAT_MODEL=qwen-plus          # 可选
    DASHSCOPE_CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  # 可选

运行：
    python memory_extract.py
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import httpx

load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# LLM 工具
# ---------------------------------------------------------------------------
def default_llm() -> ChatOpenAI:
    """对话模型：固定走阿里云百炼（勿混用 .env 中的 OPENAI_BASE_URL）。"""
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("请在 .env 中设置 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        temperature=0,
        base_url=(
            os.getenv("DASHSCOPE_CHAT_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/"),
        api_key=api_key,
        http_client=httpx.Client(verify=False),
    )


def _llm_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def _parse_json_from_llm(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM 返回为空，无法解析 JSON")
    candidates = [raw]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    brace = re.search(r"\{[\s\S]*\}", raw)
    if brace:
        candidates.append(brace.group(0))
    last_err: Optional[Exception] = None
    for piece in candidates:
        try:
            obj = json.loads(piece)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"无法从 LLM 输出解析 JSON: {last_err}")


# ---------------------------------------------------------------------------
# 记忆抽取器（工程侧 JSON 抽取，与完整链路 ingest 使用同一结构）
# ---------------------------------------------------------------------------
class MemoryExtractor:
    """从原始对话/文本抽取 summary、key_points、importance。"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def _invoke_json(self, prompt: str, *, retries: int = 2) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            p = (
                prompt
                if attempt == 0
                else prompt.rstrip() + "\n\n请只输出一个合法 JSON 对象，不要 markdown、不要解释。"
            )
            try:
                return _parse_json_from_llm(_llm_text(self.llm.invoke(p)))
            except Exception as e:
                last_err = e
        raise ValueError(str(last_err))

    @staticmethod
    def _is_trivial(content: str) -> bool:
        trivial_keywords = ["你好", "再见", "谢谢", "嗯", "哦", "好的", "哈哈"]
        return any(kw in content for kw in trivial_keywords) and len(content) < 20

    def _fallback_extract(self, content: str, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        snippet = content.strip().replace("\n", " ")
        summary = snippet[:120] + ("…" if len(snippet) > 120 else "")
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": summary,
            "key_points": [snippet] if snippet else [],
            "importance": float((metadata or {}).get("importance", 0.5)),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def extract(self, content: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self._is_trivial(content):
            return {
                "id": str(uuid.uuid4()),
                "content": content,
                "summary": content,
                "key_points": [],
                "importance": 0.1,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

        prompt = f"""请从以下内容中提取核心记忆点，并评估其重要性：
{content}

只输出一个 JSON 对象：
{{
  "summary": "一句话总结",
  "key_points": ["要点1"],
  "importance": 0.8,
  "reason": "简短说明为什么给这个分数"
}}

重要性评分标准 (0.0-1.0)：
- 1.0：硬性约束（过敏、证件号、密码）、核心身份。
- 0.8：强烈偏好（必须住海景房）、长期计划。
- 0.5：一般事实（今天吃了什么）、临时兴趣。
- 0.1：纯闲聊、问候。
"""
        try:
            extracted = self._invoke_json(prompt)
        except Exception:
            return self._fallback_extract(content, metadata)

        imp = float(extracted.get("importance", 0.5))
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": str(extracted.get("summary", content)),
            "key_points": list(extracted.get("key_points") or []),
            "importance": max(0.0, min(1.0, imp)),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }


# ---------------------------------------------------------------------------
# 思维链式抽取提示词（课堂展示用，先分析再 JSON）
# ---------------------------------------------------------------------------
MEMORY_EXTRACT_PROMPT = """在把下面这段用户对话写入长期记忆库之前，请先完成「记忆抽取预分析」。
请像整理课堂笔记一样思考，再给出结构化结果。知无不言，但只保留值得长期记住的内容。

【用户对话片段】
{dialogue}

请严格按以下四个标题逐条分析（先写分析，最后附 JSON，不要省略标题）：

1. 明确事实（日期、地点、金额、身份等可验证信息）
2. 偏好与习惯（喜欢/讨厌/要求，主观但长期有效）
3. 约束与禁忌（过敏、预算上限、硬性要求等，违反会出问题）
4. 应忽略的噪音（纯问候、无信息量的附和，不必入库）

分析写完后，在文末单独给出且仅给出一个 JSON 对象（不要 markdown 代码块），字段如下：
{{
  "summary": "一句话总结整条记忆",
  "key_points": ["原子要点1", "原子要点2"],
  "memory_type": "fact 或 preference 或 plan 或 constraint 或 chitchat",
  "importance": 0.0到1.0之间的数字,
  "reason": "一句话说明 importance 为何如此打分"
}}

importance 参考：1.0=硬性约束/过敏；0.8=强烈偏好或长期计划；0.5=一般事实；0.1=闲聊。
"""

EXTRACT_DEMO_DIALOGUE = """
user: 我打算下个月 3 月 15 日到 20 日去三亚，玩 5 天 4 夜。
user: 酒店一定要海景房，每晚别超过 800，必须有 WiFi 和空调，不要青旅。
user: 我是素食主义者，不吃海鲜，也不能吃辣，对辣椒过敏。
assistant: 好的，我都记下了。
""".strip()


def run_memory_extract_demo(dialogue: str | None = None) -> None:
    """运行记忆抽取演示：思维链预分析 + 工程侧结构化抽取。"""
    dialogue = (dialogue or EXTRACT_DEMO_DIALOGUE).strip()

    print("=" * 70)
    print("记忆抽取演示 - Memory Extraction（原子事实抽取）")
    print("=" * 70)
    print("\n【说明】仅演示抽取，不入库、不检索、不合并。")
    print("  传统抽取=原句照抄 | 本演示=summary + key_points + importance\n")
    print("【输入对话】")
    print(dialogue)
    print("\n" + "-" * 70)

    template = PromptTemplate.from_template(MEMORY_EXTRACT_PROMPT)
    full_prompt = template.format(dialogue=dialogue)

    print("【发给大模型的完整提示词】\n")
    print(full_prompt)
    print("\n" + "-" * 70)

    llm = default_llm()
    print("\n【模型输出：抽取预分析 + JSON】\n")
    response = llm.invoke([HumanMessage(content=full_prompt)])
    print(response.content)

    print("\n" + "-" * 70)
    print("【工程侧：MemoryExtractor.extract → 可直接用于 ingest 的结构】\n")
    extractor = MemoryExtractor(llm)
    structured = extractor.extract(dialogue, metadata={"source": "memory_extract_demo"})
    print(
        json.dumps(
            {
                "summary": structured["summary"],
                "key_points": structured["key_points"],
                "importance": structured["importance"],
                "memory_type_hint": "业务层传入，如 travel_plan / diet / lodging_preference",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    run_memory_extract_demo()


if __name__ == "__main__":
    main()

#输出的内容
# ======================================================================
# 【发给大模型的完整提示词】
# ----------------------------------------------------------------------
# 在我将这段对话存入长期记忆库之前，请先认真回答以下预调查问题，务必做到知无不言、言无不尽。请运用你的全部知识储备，像整理课堂笔记一样思考。

# 用户输入对话：

# user: 我打算下个月 3 月 15 日到 20 日去三亚，玩 5 天 4 夜。
# user: 酒店一定要海景房，每晚别超过 800，必须有 WiFi 和空调，不要青旅。
# user: 我是素食主义者，不吃海鲜，也不能吃辣，对辣椒过敏。
# assistant: 好的，我都记下了。

# 以下是预调查问题：

#     1. 请列出请求中明确给出的任何具体事实或数据（如日期、地点、金额）。
#     2. 请列出用户的偏好与习惯（喜欢/讨厌/要求，主观但长期有效）。
#     3. 请列出约束与禁忌（过敏、预算上限、硬性要求等，违反会出问题）。
#     4. 请列出应忽略的噪音（纯问候、无信息量的附和，不必入库）。

# 在回答此调查时，请记住，“事实”通常是具体的名称、日期、统计数据等。您的回答应使用以下标题：

#     1. 已给出或已验证的事实
#     2. 偏好与习惯
#     3. 约束与禁忌
#     4. 应忽略的噪音

# 在完成上述分析后，请在文末单独给出且仅给出一个 JSON 对象（不要 markdown 代码块），字段如下：
# {
#   "summary": "一句话总结整条记忆",
#   "key_points": ["原子要点1", "原子要点2"],
#   "memory_type": "fact 或 preference 或 plan 或 constraint",
#   "importance": 0.0到1.0之间的数字,
#   "reason": "一句话说明 importance 为何如此打分"
# }

# importance 参考：1.0=硬性约束/过敏；0.8=强烈偏好或长期计划；0.5=一般事实；0.1=闲聊。


# ----------------------------------------------------------------------
# 【模型输出：抽取预分析 + JSON】

# 1. 已给出或已验证的事实  
# - 出行时间：3月15日至3月20日（共5天4夜）  
# - 目的地：三亚  
# - 住宿要求：海景房、每晚≤800元、含WiFi、含空调、非青旅  
# - 饮食身份：素食主义者  
# - 明确禁食项：海鲜、辣椒（因过敏）  

# 2. 偏好与习惯  
# - 偏好海景房（非可选，而是“一定要”）  
# - 偏好有稳定网络（明确要求WiFi）  
# - 偏好基础舒适型住宿（强调空调、排除青旅，隐含对私密性、卫生、独立空间的重视）  
# - 长期饮食模式为素食主义（非临时选择，属身份型习惯）  

# 3. 约束与禁忌  
# - 辣椒过敏（医学级禁忌，接触即可能引发生理反应，属最高优先级安全约束）  
# - 严禁食用海鲜（虽未说明是否过敏，但结合“素食主义者”身份及“不能吃”的绝对表述，属不可妥协的饮食边界）  
# - 住宿预算硬上限：800元/晚（“别超过”为明确上限，非建议值）  
# - 住宿类型硬性排除：青旅（“不要青旅”为明确否定项）  
# - 必须配备：WiFi、空调（二者均以“必须有”表述，属功能性刚性需求）  

# 4. 应忽略的噪音  
# - “我打算……”中的“打算”属计划动词，但后续已给出精确日期和条件，故“打算”本身无信息增量，不单独保留  
# - “好，我都记下了。”——纯响应性附和，无新事实、偏好、约束，无入库价值  

# {
#   "summary": "用户计划3月15–20日赴三亚5天4夜，需预订≤800元/晚的非青旅海景房（含WiFi与空调），全程严格素食且绝对禁食海鲜与辣椒（后者系过敏）。",
#   "key_points": ["3月15–20日三亚5天4夜行程", "海景房+WiFi+空调+≤800元/晚+非青旅", "严格素食主义", "禁食海鲜", "辣椒过敏"],
#   "memory_type": "constraint",
#   "importance": 1.0,
#   "reason": "辣椒过敏是危及健康的安全红线，叠加海鲜禁令与素食身份构成不可协商的饮食铁律；住宿多项要求（价格上限、房型、设施、类型）均为明确否定式指令，共同构成高刚性执行约束。"
# }

# ======================================================================
# 演示结束。在实际工程中，我们会解析末尾的 JSON 并存入 Chroma。

