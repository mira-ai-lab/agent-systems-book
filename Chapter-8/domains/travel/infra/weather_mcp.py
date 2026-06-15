"""WeatherAPI.com MCP 客户端（weatherapi-mcp / npx stdio）。"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from agent_framework.config import load_project_dotenv

_MCP_LOCK = threading.Lock()
_SESSION: Optional["WeatherMcpSession"] = None
_MCP_MAX_ATTEMPTS = 2  # 单次失败后 reset session 并重试，不永久禁用 MCP
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
        raise RuntimeError("weatherapi-mcp 进程已关闭")
    return json.loads(line)


def _npx_command() -> list[str]:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise RuntimeError("未找到 npx，请先安装 Node.js (https://nodejs.org)")
    return [npx, "-y", "weatherapi-mcp"]


class WeatherMcpSession:
    """复用单个 npx weatherapi-mcp 子进程（stdio MCP）。"""

    def __init__(self, api_key: str) -> None:
        env = os.environ.copy()
        env["WEATHERAPI_KEY"] = api_key
        env.setdefault("PYTHONUTF8", "1")
        cmd = _npx_command()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=_MCP_ENCODING,
            errors="replace",
            env=env,
            bufsize=0,
        )
        threading.Thread(
            target=_drain_stderr,
            args=(self._process,),
            daemon=True,
        ).start()
        deadline = time.monotonic() + 8.0
        init: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("weatherapi-mcp 启动失败（npx 进程已退出，请检查 Node.js 与包名 weatherapi-mcp）")
            try:
                init = self.request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "travel-multi-agent-weather", "version": "1.0.0"},
                    },
                )
                break
            except (RuntimeError, json.JSONDecodeError, OSError):
                time.sleep(0.5)
        else:
            raise RuntimeError("weatherapi-mcp 初始化超时")
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
            "data_source": "weatherapi-mcp/current",
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
        "data_source": f"weatherapi-mcp/{tool}",
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


def fetch_weather_forecast_via_mcp_sync(city: str, days: int = 7) -> Dict[str, Any]:
    """同步调用 MCP get_forecast，返回多日预报列表。"""
    n_days = max(1, min(int(days or 7), 14))
    try:
        with _MCP_LOCK:
            session = _get_session()
            result = session.call_tool("get_forecast", {"q": city.strip(), "days": n_days})
        payload = _extract_tool_json(result)
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        location = payload.get("location") or {}
        loc_name = location.get("name") or city
        daily: List[Dict[str, Any]] = []
        for day in (payload.get("forecast") or {}).get("forecastday") or []:
            if not isinstance(day, dict):
                continue
            summary = day.get("day") or {}
            condition = (summary.get("condition") or {}).get("text") or "未知"
            daily.append({
                "date": day.get("date"),
                "condition": condition,
                "temp_high_c": summary.get("maxtemp_c"),
                "temp_low_c": summary.get("mintemp_c"),
                "avg_humidity": summary.get("avghumidity"),
                "daily_chance_of_rain": summary.get("daily_chance_of_rain"),
            })
        return {
            "city": loc_name,
            "days": len(daily),
            "forecasts": daily,
            "data_source": "weatherapi-mcp/get_forecast",
        }
    except Exception:
        with _MCP_LOCK:
            _reset_session()
        raise


def _mcp_enabled() -> bool:
    if os.getenv("WEATHER_USE_MCP", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool((os.getenv("WEATHERAPI_KEY") or "").strip())


async def _run_mcp_with_retry(sync_fn, *args: Any) -> Optional[Dict[str, Any]]:
    """MCP 调用失败时 reset session 并重试，不永久禁用后续请求。"""
    last_exc: Optional[Exception] = None
    for attempt in range(_MCP_MAX_ATTEMPTS):
        try:
            return await asyncio.to_thread(sync_fn, *args)
        except Exception as exc:
            last_exc = exc
            print(
                f"[weather-Mcp] 查询失败 (attempt {attempt + 1}/{_MCP_MAX_ATTEMPTS}): {exc}",
                flush=True,
            )
            with _MCP_LOCK:
                _reset_session()
    if last_exc is not None:
        print(f"[weather-Mcp] 放弃 MCP，由调用方回退其他数据源: {last_exc}", flush=True)
    return None


async def fetch_weather_forecast_via_mcp(city: str, days: int = 7) -> Optional[Dict[str, Any]]:
    """多日预报；失败返回 None。"""
    if not _mcp_enabled():
        return None
    return await _run_mcp_with_retry(fetch_weather_forecast_via_mcp_sync, city, days)


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
    """优先走 weatherapi-mcp；失败返回 None，由调用方回退高德/wttr.in。"""
    if not _mcp_enabled():
        return None
    return await _run_mcp_with_retry(fetch_weather_via_mcp_sync, city, norm_date)


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
