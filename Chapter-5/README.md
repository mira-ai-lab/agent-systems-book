# Chapter 5: Tool Calling and External Information Retrieval

Author: zhanghong, liuhui · Last updated: 2026-11-17

This chapter builds reusable map-backed POI services and then wraps hotel search as a LangChain tool and agent.

## Why apply for a map key

Apply for an Amap or Baidu map key so the agent can query a real hotel candidate list instead of mock data. Only one is needed.

| Provider | Purpose | Interface | Endpoint | Docs | Apply for a key |
| --- | --- | --- | --- | --- | --- |
| Amap | Query hotel, restaurant, and attraction POIs | Web Service API "Search POI 2.0" text search | `https://restapi.amap.com/v5/place/text` | [docs](https://lbs.amap.com/api/webservice/guide/api-advanced/newpoisearch) | [apply](https://console.amap.com/dev/key/app) |
| Baidu | Query hotel, restaurant, and attraction POIs | WebService API Place API 2.0 | `https://api.map.baidu.com/place/v2/search` | [docs](https://lbsyun.baidu.com/docs/webapi?title=placev3/guide/webservice-placeapiV3/interfaceDocumentV2) | [apply](https://lbsyun.baidu.com/apiconsole/key) |

## Environment variables

Configure these in the repo root `.env`:

```text
DASHSCOPE_API_KEY=...   # LLM (or OPENAI_API_KEY)
AMAP_KEY=...            # Amap Web-service key (either this or Baidu)
BAIDU_MAP_AK=...        # Baidu server-side AK (either this or Amap)
# BAIDU_MAP_SK=...      # Optional; Baidu SN verification only
```

## Call chain

`Baidu/Amap HTTP API → map_api.py → hotel_search.py → @tool recommend_hotel → create_agent → run_hotel_agent`

`restaurant_search.py` and `attraction_search.py` expose the same map-backed search pattern for restaurant and classic scenic POIs. They are reused by Chapter 6 as ordinary Python tools.

| Module | Responsibility |
| --- | --- |
| `map_api.py` | Raw Baidu/Amap POI requests and common POI normalization |
| `hotel_search.py` | Hotel keyword construction, normalized hotel search, and hotel validation |
| `restaurant_search.py` | Restaurant keyword construction, normalized restaurant search, and validation |
| `attraction_search.py` | Attraction/classic-POI search and validation |

## Run

```powershell
uv pip install -r Chapter-5/requirements.txt
```

Open `hotel_recommendation.ipynb`, select the repo-root `.venv` kernel, then run the cells in order.

---

# 第五章：工具调用与外部信息获取

作者：zhanghong, liuhui · 更新时间：2026-11-17

本章先构建可复用的地图 POI 服务，再以酒店搜索为例连接到 LangChain 工具和 Agent。

## 为什么要申请地图 Key

申请高德或百度地图 Key，是为了让 Agent 调用工具时真实查询酒店候选列表，而不是使用模拟数据。只需申请其中一个即可。

| 供应商 | 用途 | 接口 | 服务地址 | 文档 | 申请 Key |
| --- | --- | --- | --- | --- | --- |
| 高德 | 查询酒店、餐馆和景点 POI | Web服务 API「搜索POI 2.0」关键字搜索 | `https://restapi.amap.com/v5/place/text` | [docs](https://lbs.amap.com/api/webservice/guide/api-advanced/newpoisearch) | [apply](https://console.amap.com/dev/key/app) |
| 百度 | 查询酒店、餐馆和景点 POI | WebService API「地点检索2.0」 | `https://api.map.baidu.com/place/v2/search` | [docs](https://lbsyun.baidu.com/docs/webapi?title=placev3/guide/webservice-placeapiV3/interfaceDocumentV2) | [apply](https://lbsyun.baidu.com/apiconsole/key) |

## 环境变量

在仓库根目录 `.env` 中配置：

```text
DASHSCOPE_API_KEY=...   # 大模型（或 OPENAI_API_KEY）
AMAP_KEY=...            # 高德 Web 服务 Key（与百度二选一）
BAIDU_MAP_AK=...        # 百度服务端 AK（与高德二选一）
# BAIDU_MAP_SK=...      # 可选，仅百度 SN 校验需要
```

## 调用链

`百度/高德 HTTP API → map_api.py → hotel_search.py → @tool recommend_hotel → create_agent → run_hotel_agent`

`restaurant_search.py` 与 `attraction_search.py` 提供餐馆和经典景点 POI 的同类搜索能力，第 6 章直接复用这些普通 Python 工具。

| 模块 | 职责 |
| --- | --- |
| `map_api.py` | 百度/高德原始 POI 请求和通用 POI 归一化 |
| `hotel_search.py` | 酒店搜索词构造、标准化酒店搜索和有效性校验 |
| `restaurant_search.py` | 餐馆搜索词构造、标准化餐馆搜索和有效性校验 |
| `attraction_search.py` | 景点/经典 POI 搜索和有效性校验 |

## 运行

```powershell
uv pip install -r Chapter-5/requirements.txt
```

打开 `hotel_recommendation.ipynb`，选择根目录 `.venv` 内核后按顺序执行。
