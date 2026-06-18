"""领域插件注册表：entry_points 发现 + 运行时 register_domain。"""

from __future__ import annotations

from typing import Dict, List

from agent_framework.domain.plugin import DomainPlugin

_plugins: Dict[str, DomainPlugin] = {}
_discovered = False


def register_domain(plugin: DomainPlugin) -> None:
    """注册或覆盖一个领域插件。"""
    _plugins[plugin.name] = plugin


def get_domain_plugin(name: str) -> DomainPlugin:
    """按名称获取插件；未知领域抛出 KeyError。"""
    ensure_domains_loaded()
    key = (name or "").strip()
    if key not in _plugins:
        available = ", ".join(sorted(_plugins)) or "(none)"
        raise KeyError(
            f"未知领域 '{name}'，已注册: {available}。"
            " 请执行: pip install -e domains/ 或在仓库根目录运行 scripts/install_dev.ps1"
        )
    return _plugins[key]


def list_domains() -> List[Dict[str, str]]:
    """返回已注册领域摘要（供 API / CLI 使用）。"""
    ensure_domains_loaded()
    return [
        {
            "name": plugin.name,
            "display_name": plugin.display_name or plugin.name,
            "modes": list(plugin.supported_modes),
            "is_sample": plugin.is_sample,
            "recommended": not plugin.is_sample,
            "recommended_profile": "auto",
        }
        for plugin in sorted(
            _plugins.values(),
            key=lambda p: (p.is_sample, p.name),
        )
    ]


def clear_domains() -> None:
    """测试用：清空注册表（含发现标记）。"""
    global _discovered
    _plugins.clear()
    _discovered = False


def ensure_domains_loaded() -> None:
    """通过 entry_points（agent_platform.domains）惰性发现插件。"""
    global _discovered
    if _discovered:
        return
    from agent_framework.domain.entrypoint_loader import (
        load_dev_fallback_plugins,
        load_plugins_from_entrypoints,
    )

    load_plugins_from_entrypoints()
    if not _plugins:
        load_dev_fallback_plugins()
    _discovered = True


# 兼容旧名
ensure_builtin_domains = ensure_domains_loaded
