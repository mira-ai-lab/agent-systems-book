"""
Hotel Recommendation Agent: recommend hotels and accommodation based on city and optional constraints.
Tool column in the benchmark image: 百度地图（已接入 Web 服务 Place API，未配置则回退到 stub）
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

try:
    from logging_utils import get_agent_logger
    from travel_common import fetch_hotels_from_api, norm_text, require_non_empty
except ModuleNotFoundError:  # pragma: no cover
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from logging_utils import get_agent_logger
    from travel_common import fetch_hotels_from_api, norm_text, require_non_empty


def _bootstrap_env() -> None:
    """加载本地 .env 与书根目录 .env（含 BAIDU_MAP_AK / DASHSCOPE_*）。"""
    agent_dir = Path(__file__).resolve().parent
    book_root = agent_dir.parent.parent.parent
    ch6_dir = book_root / "Chapter-6"
    for p in (agent_dir, ch6_dir):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    load_dotenv(agent_dir / ".env", override=False)
    load_dotenv(book_root / ".env", override=False)


_bootstrap_env()
memory = MemorySaver()
logger = get_agent_logger("HotelRecommendationAgent")


@dataclass
class AgentStreamItem:
    content: str
    is_task_complete: bool = False
    require_user_input: bool = False


def _valid_hotel_poi(h: Dict[str, Any]) -> bool:
    name = (h.get("name") or "").strip()
    if not name:
        return False
    if name.endswith("市") and not h.get("address") and not h.get("district"):
        return False
    return True


def _chunk_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "".join(parts)
    return str(content)


def _message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        return _chunk_text(message.get("content"))
    return _chunk_text(getattr(message, "content", message))


def _create_chat_model() -> ChatOpenAI:
    api_key = (
        os.getenv("CHAT_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("请设置 CHAT_API_KEY 或 DASHSCOPE_API_KEY")

    base_url = (
        os.getenv("CHAT_ENDPOINT")
        or os.getenv("DASHSCOPE_CHAT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = os.getenv("DEPLOYMENT_NAME") or os.getenv("DASHSCOPE_CHAT_MODEL") or "qwen-plus"
    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() not in ("0", "false", "no")

    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        request_timeout=60,
        max_retries=2,
        streaming=True,
        http_client=httpx.Client(verify=ssl_verify),
    )


@tool
async def recommend_hotel(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """查酒店列表。preferences 传区名/景点/品牌，或主观偏好（安静、亲子等，走语义检索）。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    pref = norm_text(preferences)

    hotels: List[Dict[str, Any]] = []
    source = "none"
    search_query = None
    search_tag = None
    baidu_filter = None
    region_limit = None
    try:
        res = await fetch_hotels_from_api(
            city,
            limit=10,
            keyword=pref or None,
            budget_cny_per_night_max=budget_cny_per_night_max,
        )
        if not res.get("error"):
            source = res.get("data_source", "api")
            search_query = res.get("search_query")
            search_tag = res.get("search_tag")
            baidu_filter = res.get("baidu_filter")
            region_limit = res.get("region_limit")
            for h in res.get("hotels") or []:
                if isinstance(h, dict):
                    item = {k: v for k, v in h.items() if k != "raw"}
                    if _valid_hotel_poi(item):
                        if budget_cny_per_night_max:
                            price = h.get("avg_price_cny")
                            if price and price > budget_cny_per_night_max:
                                continue
                        hotels.append(item)
    except Exception as exc:
        return {"error": f"hotel_query_failed: {exc}"}

    return {
        "city": city,
        "search_query": search_query,
        "search_tag": search_tag,
        "baidu_filter": baidu_filter,
        "region_limit": region_limit,
        "preferences": pref or None,
        "budget_cny_per_night_max": budget_cny_per_night_max,
        "hotels": hotels,
        "count": len(hotels),
        "data_source": source,
        "note": "请根据用户要求的数量从 hotels 中推荐（默认 1 家；用户说 N 家则推荐 N 家，最多 5 家），每家说明名称、地址、价格/评分与理由。",
    }


class HotelRecommendationAgent:
    SYSTEM_INSTRUCTION = """你是酒店推荐助手，只能通过工具 recommend_hotel 查酒店。
规则：
1. 从用户话里提取 city（必填）；预算写入 budget_cny_per_night_max。
2. preferences 参数：
   - 地图关键词：区名、景点、地标（如 古城、云冈、平城区）→ Place v2 检索
   - 主观偏好：安静、亲子、性价比等 → 写入 preferences，会走 Place Pro 语义检索（如「亲子友好酒店」）
3. 推荐数量：用户明确说「N 个/家」时，从 hotels 里推荐 N 家（最多 5 家）；未指定数量时默认只推荐 1 家最合适的。
4. 每家需包含：名称、地址/位置、参考价格、评分（如有）、1 句推荐理由；按匹配度排序。
5. 入住/退房日期若用户提供，使用 YYYY-MM-DD 格式并在回复中确认。
6. 若 hotels 为空或不足用户要求数量，如实说明并给出已有选项或调整建议。
7. 非酒店相关问题，回复：我只能协助酒店推荐。
"""

    def __init__(self):
        self.model = _create_chat_model()
        self.tools = [recommend_hotel]
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            system_prompt=self.SYSTEM_INSTRUCTION,
            checkpointer=memory,
        )

    async def stream(self, query: str, context_id: str = "default") -> AsyncIterable[AgentStreamItem]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}
        streamed = False
        try:
            t0 = time.perf_counter()
            first_token_at: float | None = None
            llm_call_t0: float | None = None
            tool_t0_by_run_id: Dict[str, float] = {}
            async for event in self.agent.astream_events(inputs, config, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_start":
                    llm_call_t0 = time.perf_counter()
                    logger.debug("llm_start context_id=%s", context_id)
                elif kind == "on_chat_model_stream":
                    text = _chunk_text(event["data"]["chunk"].content)
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            logger.info(
                                "first_token context_id=%s first_token_ms=%s",
                                context_id,
                                int((first_token_at - t0) * 1000),
                            )
                        streamed = True
                        yield AgentStreamItem(content=text)
                elif kind == "on_chat_model_end":
                    if llm_call_t0 is not None:
                        logger.debug(
                            "llm_end context_id=%s llm_ms=%s",
                            context_id,
                            int((time.perf_counter() - llm_call_t0) * 1000),
                        )
                    if not streamed:
                        text = _message_text(event.get("data", {}).get("output"))
                        if text:
                            streamed = True
                            yield AgentStreamItem(content=text)
                    llm_call_t0 = None
                elif kind == "on_tool_start":
                    tool_name = event.get("name") or "unknown_tool"
                    run_id = str(event.get("run_id") or "")
                    if run_id:
                        tool_t0_by_run_id[run_id] = time.perf_counter()
                    logger.info("tool_start context_id=%s tool=%s", context_id, tool_name)
                elif kind == "on_tool_end":
                    tool_name = event.get("name") or "unknown_tool"
                    run_id = str(event.get("run_id") or "")
                    t_start = tool_t0_by_run_id.pop(run_id, None) if run_id else None
                    tool_ms = int((time.perf_counter() - t_start) * 1000) if t_start is not None else None
                    logger.info("tool_end context_id=%s tool=%s tool_ms=%s", context_id, tool_name, tool_ms)
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    if not streamed:
                        out = event.get("data", {}).get("output") or {}
                        messages = out.get("messages") if isinstance(out, dict) else None
                        if messages:
                            text = _message_text(messages[-1])
                            if text:
                                streamed = True
                                yield AgentStreamItem(content=text)
                    logger.info(
                        "agent_done context_id=%s total_ms=%s streamed=%s",
                        context_id,
                        int((time.perf_counter() - t0) * 1000),
                        streamed,
                    )
                    yield AgentStreamItem(content="", is_task_complete=True)
                    break
        except Exception as e:
            logger.exception("agent_error context_id=%s err=%s", context_id, e)
            yield AgentStreamItem(content=f"Error: {str(e)}", is_task_complete=True)

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]


if __name__ == "__main__":
    import asyncio

    # 检索策略：
    # - 「近古城」→ Place v2: query=古城, tag=酒店
    # - 「安静、亲子」→ Place Pro 多维检索: query=安静亲子酒店（自然语言，非词表映射）
    EXAMPLE_QUERIES = [
        "推荐一个大同的酒店，预算不超过500/晚，2026-06-04入住 2026-06-05退房",
        "推荐一个大同近古城的酒店，预算不超过500/晚，2026-06-04入住 2026-06-05退房",
        "推荐一个大同安静、适合亲子的酒店，预算不超过500/晚",
    ]

    async def _test():
        agent = HotelRecommendationAgent()
        query = EXAMPLE_QUERIES[1]  # 改索引 0/1/2 对比不同偏好
        print(f"用户: {query}\n")
        async for item in agent.stream(query, "test_hotel_1"):
            if item.content:
                print(item.content, end="", flush=True)
        print("\nDone.")

    asyncio.run(_test())
