"""Shared fact-survey prompt, data model, and parser.

The prompt separates known facts, facts to verify, facts to derive, and
educated guesses while grounding relative dates to an explicit current date.

The Chapter-2 teaching notebook remains self-contained and does not import this
module; later chapters use it as the reusable implementation.

:author: zhanghong, liuhui
:date: 2026-11-18
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from llm import parse_json_response


class FactSurvey(BaseModel):
    """Structured result of the four-part fact survey."""

    given_facts: List[str] = Field(default_factory=list)
    facts_to_lookup: List[str] = Field(default_factory=list)
    facts_to_derive: List[str] = Field(default_factory=list)
    educated_guesses: List[str] = Field(default_factory=list)
    raw_text: str = ""


def parse_fact_survey(text: str) -> FactSurvey:
    """Parse model output into the declared ``FactSurvey`` structure."""
    payload = parse_json_response(text)
    if not isinstance(payload, dict):
        raise ValueError("Fact survey response must be a JSON object")
    return FactSurvey.model_validate({**payload, "raw_text": text})


FACTS_PROMPT = """在处理用户任务之前，请先完成以下信息分析。回答应准确、完整，并明确区分已知信息、待查证信息、推导结果与不确定性判断。

当前日期：{today}
- 遇到“今天”“明天”“后天”“下周”等相对日期时，以 {today} 为基准换算。
- 相对日期不要只保留原表达，应在对应事实项中写出明确的 YYYY-MM-DD 日期，可同时保留原表达以便阅读。
- 不得使用过时年份作为示例日期。

用户输入任务：

{task}

以下是预调查问题：

    1. 请列出请求中明确给出的任何具体事实或数据。可能没有任何此类信息。
    2. 请列出可能需要查阅的任何事实，以及具体可以在哪里找到这些信息。在某些情况下，请求本身会提及权威来源。
    3. 请列出可能需要推导的任何事实（例如，通过逻辑演绎、模拟或计算得出）。
    4. 请列出从记忆中回忆出的任何事实、直觉、经过充分推理的猜测等。

请只输出一个 JSON 对象，不要输出 Markdown 代码块或解释文字。字段定义如下：
{{
  "given_facts": ["已给出或已验证的事实"],
  "facts_to_lookup": ["需要查阅的事实"],
  "facts_to_derive": ["需要推导的事实"],
  "educated_guesses": ["有根据的猜测"]
}}
无法确定的数组字段返回空数组 []，不得编造。
"""
