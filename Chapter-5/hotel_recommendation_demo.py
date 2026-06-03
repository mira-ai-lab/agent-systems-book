"""
酒店推荐 Agent 示例 - 演示「大模型自动调用工具」

读者只需抓住一条链路：
  用户提问 → Agent（大模型）→ 自动调用 recommend_hotel 工具 → 百度地图查酒店 → 模型组织回答

依赖（写在书里的 .env 示例即可）：
  DASHSCOPE_API_KEY=...        # 大模型
  BAIDU_MAP_AK=...             # 百度地图 Place API（没有则回退到内置示例数据）

"""

import asyncio
import json
import os
import sys
import urllib
from datetime import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, List, Optional, Tuple


# 为了能 import 项目里的 travel_common（地图 API、pick_one 等小函数）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "enterprise_bench"))

_DOTENV_LOADED = False
_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _baidu_result_to_common_poi(item: Dict[str, Any]) -> Dict[str, Any]:
    loc = item.get("location") or {}
    lat = loc.get("lat")
    lng = loc.get("lng")
    detail = item.get("detail_info") or {}
    # Baidu detail_info price / overall_rating are strings sometimes
    try:
        rating = float(detail.get("overall_rating")) if detail.get("overall_rating") is not None and str(detail.get("overall_rating")).strip() else None
    except Exception:
        rating = None
    price = detail.get("price")
    try:
        avg_price = int(float(str(price).replace("￥", "").replace("元", "").strip())) if price is not None and str(price).strip() else None
    except Exception:
        avg_price = None

    location_str = None
    if lng is not None and lat is not None:
        location_str = f"{lng},{lat}"
    return {
        "name": item.get("name"),
        "district": item.get("area") or item.get("city") or item.get("address"),
        "address": item.get("address"),
        "tel": item.get("telephone"),
        "location": location_str,
        "rating": rating,
        "avg_price_cny": avg_price,
        "type": detail.get("type") or item.get("tag") or item.get("detail_info", {}).get("tag") or item.get("type"),
        "raw": item,
    }

def _md5_hex(s: str) -> str:
    import hashlib as _hashlib

    return _hashlib.md5(s.encode("utf-8")).hexdigest()
def _baidu_sn(path: str, params_in_order: List[Tuple[str, str]], sk: str) -> str:
    """
    Best-effort SN signing for Baidu WebService APIs.
    Reference: Baidu SN signature mechanism (requires ordered params).
    """
    qs = urllib.parse.urlencode(params_in_order, safe="|,:", quote_via=urllib.parse.quote)
    whole = f"{path}?{qs}{sk}"
    return _md5_hex(urllib.parse.quote(whole, safe=""))

async def baidu_place_v2_search(
    *,
    query: str,
    region: str,
    tag: Optional[str] = None,
    scope: int = 2,
    page_size: int = 10,
    page_num: int = 0,
    filter_: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Baidu Map Place API v2 search.
    Docs: https://lbsyun.baidu.com/docs/webapi?title=placev3%2Fguide%2Fwebservice-placeapiV3%2FinterfaceDocumentV2
    """
    ensure_project_dotenv_loaded()
    ak = norm_text(os.getenv("BAIDU_MAP_AK"))
    if not ak:
        return {"error": "BAIDU_MAP_AK not set"}
    sk = norm_text(os.getenv("BAIDU_MAP_SK"))

    url = "https://api.map.baidu.com/place/v2/search"
    # Keep param order stable for SN
    params_in_order: List[Tuple[str, str]] = [
        ("query", norm_text(query)),
        ("region", norm_text(region)),
        ("output", "json"),
        ("scope", str(int(scope or 2))),
        ("page_size", str(max(1, min(int(page_size or 10), 20)))),
        ("page_num", str(max(0, int(page_num or 0)))),
        ("ak", ak),
    ]
    if tag:
        params_in_order.insert(1, ("tag", norm_text(tag)))
    if filter_:
        # Put filter close to scope as per common examples; order must match request
        params_in_order.insert(5, ("filter", norm_text(filter_)))
    if sk:
        params_in_order.append(("timestamp", str(int(time.time()))))
        sn = _baidu_sn("/place/v2/search", params_in_order, sk)
        params_in_order.append(("sn", sn))

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, params=params_in_order)
        r.raise_for_status()
        data = r.json()

    if not isinstance(data, dict):
        return {"error": "invalid_baidu_response", "raw": data}
    # status: 0 ok
    if str(data.get("status")) not in ("0", "OK", "ok"):
        return {"error": "baidu_error", "raw": data, "message": data.get("message")}
    results = data.get("results")
    if not isinstance(results, list):
        return {"error": "invalid_baidu_results", "raw": data}
    return {"results": results, "raw": data}




async def _http_get_json(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data
        return {"data": data}


async def fetch_hotels_from_api(city: str, *, limit: int = 10, keyword: Optional[str] = None) -> Dict[str, Any]:
    # Prefer Baidu Place API when configured, else fallback to AMAP.
    # keyword: 区域/地标偏好，如「黄浦区」「外滩」；会拼进 POI 搜索词（默认仅「酒店」）
    ensure_project_dotenv_loaded()
    q = norm_text(keyword) or "酒店"
    if "酒店" not in q:
        q = f"{q} 酒店"
    page_n = max(1, min(int(limit or 10), 20))
    if norm_text(os.getenv("BAIDU_MAP_AK")):
        res = await baidu_place_v2_search(
            query=q,
            region=city,
            scope=2,
            page_size=page_n,
            filter_="industry_type:hotel|sort_name:total_score|sort_rule:0",
        )
        if not res.get("error"):
            hotels = []
            for it in (res.get("results") or [])[:page_n]:
                if isinstance(it, dict):
                    hotels.append(_baidu_result_to_common_poi(it))
            return {"hotels": hotels, "data_source": "baidu_place_v2", "search_query": q}


def _project_root_dir() -> str:
    # project root is two levels up from this file's directory.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ensure_project_dotenv_loaded() -> None:
    """
    Make loading `.env` robust across IDE run/debug configurations.
    Loads `<project_root>/.env` once if present.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    root = _project_root_dir()
    dotenv_path = os.path.join(root, ".env")
    try:
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=False)
    finally:
        _DOTENV_LOADED = True

def norm_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _poi_to_hotel(poi: Dict[str, Any]) -> Dict[str, Any]:
    biz = poi.get("business") or {}
    rating = biz.get("rating")
    cost = biz.get("cost")
    try:
        rating_f = float(rating) if rating is not None and str(rating).strip() else None
    except Exception:
        rating_f = None
    try:
        cost_i = int(float(cost)) if cost is not None and str(cost).strip() else None
    except Exception:
        cost_i = None
    return {
        "name": poi.get("name"),
        "district": poi.get("adname") or poi.get("address"),
        "address": poi.get("address"),
        "tel": poi.get("tel"),
        "location": poi.get("location"),
        "rating": rating_f,
        "avg_price_cny": cost_i,
        "type": poi.get("type"),
    }

def require_non_empty(value: Optional[str], field: str) -> Tuple[bool, str]:
    if norm_text(value):
        return True, ""
    return False, f"{field} is required."

load_dotenv()
ensure_project_dotenv_loaded()

# ---------- 示例输入 ----------
USER_QUERY = "我要去大同玩三天给我推荐酒店，需要安静/近景区"

SYSTEM_INSTRUCTION = """你是酒店推荐助手，只能通过工具 recommend_hotel 查酒店。
规则：
1. 从用户话里提取 city（必填）；预算、区域、品牌等写入工具参数（供记录），工具会返回候选列表。
2. 根据工具返回的 hotels 列表，结合用户预算与偏好，由你挑选并只向用户推荐一家最合适的酒店。
3. 非酒店相关问题，回复：我只能协助酒店推荐。"""


# ---------- 第一步：定义工具（@tool 装饰器供 Agent 绑定） ----------


@tool
async def recommend_hotel(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """查询酒店候选列表（不做规则筛选），由大模型根据返回结果为用户选一家。preferences 可写区名、品牌或地标。"""
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    # 拼 POI 搜索词：有偏好用「偏好+酒店」，否则只搜「酒店」
    pref = norm_text(preferences)
    keyword = pref if pref and "酒店" in pref else (f"{pref} 酒店" if pref else "酒店")

    # 调地图 API（内部：百度优先，失败则用示例数据 HOTELS）
    candidates: List[Dict[str, Any]] = []
    source = "stub(hotels)"
    try:
        res = await fetch_hotels_from_api(city, limit=10, keyword=keyword)
        if not res.get("error"):
            candidates = list(res.get("hotels") or [])
            source = res.get("data_source", "api")
    except Exception:
        pass

    # 不做预算/区域等规则过滤，原样交给大模型挑选
    hotels: List[Dict[str, Any]] = []
    for h in candidates:
        if not isinstance(h, dict):
            continue
        item = dict(h)
        item.pop("raw", None)
        hotels.append(item)

    return {
        "city": city,
        "search_query": keyword,
        "preferences": pref or None,
        "budget_cny_per_night_max": int(budget_cny_per_night_max) if budget_cny_per_night_max else None,
        "hotels": hotels,
        "data_source": source,
        "note": "请根据用户诉求从 hotels 中挑选一家最合适的酒店向用户说明。",
    }


# ---------- 第二步：组装 Agent（模型 + 工具 + 系统提示词） ----------

memory = MemorySaver()

llm = ChatOpenAI(
    model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/"),
    http_client=httpx.Client(verify=False),
)

agent = create_agent(
    llm,
    tools=[recommend_hotel],
    system_prompt=SYSTEM_INSTRUCTION,
    checkpointer=memory,
)


# ---------- 第三步：运行——观察「模型何时调工具」 ----------


def _short_json(data: Any) -> str:
    if hasattr(data, "content"):  # LangChain 工具返回对象
        data = data.content
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data[:800]
    text = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (dict, list)) else str(data)
    return text if len(text) <= 800 else text[:800] + "…"


async def demo_direct_tool() -> None:
    """对比：不经过 Agent，程序员直接调工具。"""
    print("=" * 60)
    print("【对比】直接调用工具")
    print("=" * 60)
    result = await recommend_hotel.ainvoke({"city": "大同"})
    print(f"搜索词: {result.get('search_query')}")
    print(f"共 {len(result.get('hotels') or [])} 家候选（交给模型挑选，工具不做过滤）")
    for i, h in enumerate((result.get("hotels") or []), 1):
        print(f"  {i}. {h.get('name')} | {h.get('district') or h.get('address')}")
    print()


async def demo_agent_calls_tool() -> None:
    """正文：Agent 根据用户话自动决定调用 recommend_hotel。"""
    print("=" * 60)
    print("【正文】Agent 自动工具调用")
    print("=" * 60)
    print(f"用户: {USER_QUERY}\n")

    inputs = {"messages": [("user", USER_QUERY)]}
    config = {"configurable": {"thread_id": "book_demo"}}

    print("模型回答: ", end="", flush=True)
    async for event in agent.astream_events(inputs, config, version="v2"):
        kind = event["event"]

        # 流式输出模型文字
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)

        # 工具被调用时打印参数（书里用这两行说明「Agent 调用了工具」）
        elif kind == "on_tool_start":
            print(f"\n\n>>> 工具调用: {event.get('name')}")
            print(f">>> 参数: {_short_json(event['data'].get('input'))}")

        elif kind == "on_tool_end":
            print(f">>> 工具返回: {_short_json(event['data'].get('output'))}\n")
            print("模型继续: ", end="", flush=True)

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            break

    print("\n")


async def main() -> None:
    await demo_direct_tool()
    await demo_agent_calls_tool()


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# 【对比】直接调用工具
# ============================================================
# 搜索词: 酒店
# 共 10 家候选（交给模型挑选，工具不做过滤）
#   1. 佳园宾馆 | 云冈区
#   2. 天镇久天宾馆 | 天镇县
#   3. 大同海波诚信驿站民宿 | 平城区
#   4. 家馨宾馆 | 广灵县
#   5. 大同朴宿微澜民宿 | 平城区
#   6. 大同碧海情缘公寓 | 平城区
#   7. 微风轻语民宿 | 平城区
#   8. 闲然之家民宿 | 平城区
#   9. 日昇农家民宿 | 云州区
#   10. 玥庭兰舍民宿 | 浑源县
#
# ============================================================
# 【正文】Agent 自动工具调用
# ============================================================
# 用户: 我要去大同玩三天给我推荐酒店，需要安静/近景区
#
# 模型回答:
#
# >>> 工具调用: recommend_hotel
# >>> 参数: {
#   "city": "大同",
#   "preferences": "近景区"
# }
# >>> 工具返回: {
#   "city": "大同",
#   "search_query": "近景区 酒店",
#   "preferences": "近景区",
#   "budget_cny_per_night_max": null,
#   "hotels": [
#     {
#       "name": "浑源旨岭宜景酒店(北岳恒山景区真武庙店)",
#       "district": "浑源县",
#       "address": "恒山景区真武庙旁边",
#       "tel": "15525220428",
#       "location": "113.740723,39.668781",
#       "rating": 4.8,
#       "avg_price_cny": null,
#       "type": "hotel"
#     },
#     {
#       "name": "遇见·恒山",
#       "district": "浑源县",
#       "address": "大同市恒山北路国际绿洲·和园七号楼一单元801",
#       "tel": "13935273488",
#       "location": "113.70018,39.71848",
#       "rating": 4.8,
#       "avg_price_cny": null,
#       "type": "hotel"
#     },
#     {
#       "name": "瑞福民宿(大同古城景区店)",
#       "district": "平城区",
#       "address": "山西省大同市平城区古城街道大同古城云路街20号",
#       "tel": "15303524888",
#       "location": "113.312935,40.092551",
#       "rati…
#
# 模型继续: 根据您的需求——在大同游玩三天、偏好**安静**且**靠近景区**，我为您精选推荐：
#
# 🏨 **瑞福民宿（大同古城景区店）**
# 📍 地址：大同市平城区古城街道云路街20号（就在大同古城核心景区内）
# ✅ 优势突出：
# - 真正「近景区」：步行即达华严寺、善化寺、九龙壁、鼓楼等古城核心景点；
# - 环境安静：位于古城内静谧街巷，远离主干道喧嚣，民宿设计注重私密与舒适；
# - 品质保障：评分4.7分（高口碑），位置与体验兼顾，适合深度游览古城的三日行程。
#
# 如需协助预订或了解房型/价格，欢迎随时告诉我！
