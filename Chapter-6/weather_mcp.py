"""WeatherAPI.com MCP 客户端（weatherapi-Mcp / npx stdio）。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, Optional

from chapter6.paths import load_project_dotenv

load_project_dotenv()

_MCP_LOCK = threading.Lock()
_SESSION: Optional["WeatherMcpSession"] = None
_REQ_ID = 0
# Windows 默认 GBK 解码 MCP 的 UTF-8 JSON 会报错，必须显式 utf-8
_MCP_ENCODING = "utf-8"


def _next_req_id() -> int:
    global _REQ_ID
    _REQ_ID += 1
    return _REQ_ID


def _reset_session() -> None:
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:
            pass
        _SESSION = None


def _drain_stderr(process: subprocess.Popen) -> None:
    if not process.stderr:
        return
    for line in process.stderr:
        print(f"[weather-Mcp stderr] {line.rstrip()}", file=sys.stderr, flush=True)


def _mcp_send(process: subprocess.Popen, message: Dict[str, Any]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _mcp_read(process: subprocess.Popen) -> Dict[str, Any]:
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("weatherapi-Mcp 进程已关闭")
    return json.loads(line)


class WeatherMcpSession:
    """复用单个 npx weatherapi-Mcp 子进程（stdio MCP）。"""

    def __init__(self, api_key: str) -> None:
        env = os.environ.copy()
        env["WEATHERAPI_KEY"] = api_key
        # 避免 npx/node 在中文 Windows 下继承错误代码页
        env.setdefault("PYTHONUTF8", "1")
        self._process = subprocess.Popen(
            "npx -y weatherapi-Mcp",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=_MCP_ENCODING,
            errors="replace",
            env=env,
            bufsize=0,
            shell=sys.platform == "win32",
        )
        threading.Thread(
            target=_drain_stderr,
            args=(self._process,),
            daemon=True,
        ).start()
        time.sleep(5)
        init = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "chapter6-weather-agent", "version": "1.0.0"},
            },
        )
        if init.get("error"):
            raise RuntimeError(f"MCP initialize 失败: {init['error']}")
        self.notify("notifications/initialized")

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": _next_req_id(),
            "method": method,
            "params": params or {},
        }
        _mcp_send(self._process, payload)
        return _mcp_read(self._process)

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        notification: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        _mcp_send(self._process, notification)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        response = self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response.get("result") or response

    def close(self) -> None:
        try:
            if self._process.stdin:
                self._process.stdin.close()
            if self._process.stdout:
                self._process.stdout.close()
        except OSError:
            pass
        self._process.terminate()


def _get_session() -> WeatherMcpSession:
    global _SESSION
    api_key = (os.getenv("WEATHERAPI_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("WEATHERAPI_KEY 未配置")
    if _SESSION is None:
        _SESSION = WeatherMcpSession(api_key)
    return _SESSION


def _extract_tool_json(result: Dict[str, Any]) -> Dict[str, Any]:
    content = result.get("content") or []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text") or ""
            if text.strip():
                return json.loads(text)
    raise RuntimeError("MCP 工具未返回可解析 JSON")


def _pick_forecast_day(payload: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    forecast = payload.get("forecast") or {}
    for day in forecast.get("forecastday") or []:
        if isinstance(day, dict) and day.get("date") == target:
            return day
    return None


def _normalize_weather_payload(
    city: str,
    norm_date: str,
    payload: Dict[str, Any],
    *,
    tool: str,
) -> Dict[str, Any]:
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    location = payload.get("location") or {}
    loc_name = location.get("name") or city

    if tool == "get_current_weather":
        current = payload.get("current") or {}
        condition = (current.get("condition") or {}).get("text") or "未知"
        return {
            "city": loc_name,
            "date": norm_date,
            "forecast": {
                "condition": condition,
                "temp_c": current.get("temp_c"),
                "temp_high_c": current.get("temp_c"),
                "temp_low_c": current.get("temp_c"),
                "humidity": current.get("humidity"),
                "wind_kph": current.get("wind_kph"),
                "advice": condition,
            },
            "raw": payload,
            "data_source": "weatherapi-Mcp/current",
        }

    day = _pick_forecast_day(payload, norm_date)
    if day is None and tool == "get_history":
        day_info = (payload.get("forecast") or {}).get("forecastday") or []
        day = day_info[0] if day_info else None
    if day is None:
        raise RuntimeError(f"未找到 {norm_date} 的预报数据")

    summary = day.get("day") or {}
    condition = (summary.get("condition") or {}).get("text") or "未知"
    return {
        "city": loc_name,
        "date": norm_date,
        "forecast": {
            "condition": condition,
            "temp_high_c": summary.get("maxtemp_c"),
            "temp_low_c": summary.get("mintemp_c"),
            "avg_humidity": summary.get("avghumidity"),
            "daily_chance_of_rain": summary.get("daily_chance_of_rain"),
            "advice": condition,
        },
        "raw": payload,
        "data_source": f"weatherapi-Mcp/{tool}",
    }


def _choose_tool_and_args(city: str, norm_date: str) -> tuple[str, Dict[str, Any]]:
    target = datetime.strptime(norm_date, "%Y-%m-%d").date()
    today = date.today()
    q = city.strip()

    if target == today:
        return "get_current_weather", {"q": q}

    if target < today:
        if target < date(2010, 1, 1):
            raise RuntimeError("历史日期早于 2010-01-01，MCP 不支持")
        return "get_history", {"q": q, "dt": norm_date}

    delta_days = (target - today).days
    if delta_days <= 14:
        return "get_forecast", {"q": q, "days": min(14, delta_days + 1)}

    if 14 < delta_days <= 300:
        return "get_future_weather", {"q": q, "dt": norm_date}

    raise RuntimeError(f"日期 {norm_date} 超出 MCP 可查询范围")


def fetch_weather_via_mcp_sync(city: str, norm_date: str) -> Dict[str, Any]:
    """同步调用 MCP 查询天气（需在已有 event loop 外或 to_thread 中使用）。"""
    tool, arguments = _choose_tool_and_args(city, norm_date)
    try:
        with _MCP_LOCK:
            session = _get_session()
            result = session.call_tool(tool, arguments)
        payload = _extract_tool_json(result)
        return _normalize_weather_payload(city, norm_date, payload, tool=tool)
    except Exception:
        with _MCP_LOCK:
            _reset_session()
        raise


async def fetch_weather_via_mcp(city: str, norm_date: str) -> Optional[Dict[str, Any]]:
    """优先走 weatherapi-Mcp；失败返回 None，由调用方回退其他 API。"""
    if not (os.getenv("WEATHERAPI_KEY") or "").strip():
        return None
    try:
        return await asyncio.to_thread(fetch_weather_via_mcp_sync, city, norm_date)
    except Exception as exc:
        print(f"[weather-Mcp] 查询失败，将回退其他数据源: {exc}", flush=True)
        return None


def close_weather_mcp() -> None:
    with _MCP_LOCK:
        _reset_session()


if __name__ == "__main__":
    demo_city = "Beijing"
    demo_date = date.today().strftime("%Y-%m-%d")
    print("✅ 正在通过 MCP 查询天气…", flush=True)
    data = fetch_weather_via_mcp_sync(demo_city, demo_date)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    close_weather_mcp()
    print("🔌 MCP 已关闭", flush=True)
