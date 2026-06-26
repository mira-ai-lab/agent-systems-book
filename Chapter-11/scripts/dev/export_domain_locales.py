"""从 domains/*/prompts*.py 导出 locales/{zh,en}.json。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAIN_FIELD_MAP = {
    "travel": {
        "zh": {
            "central_agent_system": ("domains.travel.prompts", "CENTRAL_AGENT_SYSTEM_PROMPT"),
            "aggregation": ("domains.travel.prompts", "AGGREGATION_PROMPT"),
            "facts_prompt": ("domains.travel.prompts", "FACTS_PROMPT"),
            "decomposition_prompt": ("domains.travel.prompts", "PROMPT_TP_ZH"),
            "dependency_system": ("domains.travel.prompts", "DEPENDENCY_SYSTEM_PROMPT_ZH"),
            "dependency_user": ("domains.travel.prompts", "DEPENDENCY_USER_PROMPT_ZH"),
            "agent_routing": ("domains.travel.prompts", "AGENT_ROUTING_PROMPT"),
            "supervisor_system": ("domains.travel.prompts", "SUPERVISOR_SYSTEM_PROMPT"),
            "multi_task_title": ("__const__", "📋 最终旅行规划"),
            "single_task_title": ("__const__", "📋 最终回复"),
            "aggregation_skip_hint": (
                "__const__",
                "单任务查询，直接使用子智能体回复（跳过旅行规划聚合）",
            ),
            "memory_aggregation_instruction": (
                "__const__",
                "请根据用户原始请求的范围，综合子任务执行结果生成回复。"
                "严格匹配用户问题，不要添加用户未询问的内容（例如用户只问天气，不要输出行程/酒店/美食攻略）。"
                "仅当用户明确要求旅行规划时，才提供完整的多日行程方案。",
            ),
        },
        "en": {
            "central_agent_system": ("domains.travel.prompts_en", "CENTRAL_AGENT_SYSTEM_PROMPT"),
            "aggregation": ("domains.travel.prompts_en", "AGGREGATION_PROMPT"),
            "facts_prompt": ("domains.travel.prompts_en", "FACTS_PROMPT"),
            "decomposition_prompt": ("domains.travel.prompts_en", "PROMPT_TP_EN"),
            "dependency_system": ("domains.travel.prompts_en", "DEPENDENCY_SYSTEM_PROMPT_EN"),
            "dependency_user": ("domains.travel.prompts_en", "DEPENDENCY_USER_PROMPT_EN"),
            "agent_routing": ("domains.travel.prompts_en", "AGENT_ROUTING_PROMPT"),
            "supervisor_system": ("domains.travel.prompts_en", "SUPERVISOR_SYSTEM_PROMPT"),
            "multi_task_title": ("__const__", "Final Travel Plan"),
            "single_task_title": ("__const__", "Final Reply"),
            "aggregation_skip_hint": (
                "__const__",
                "Single-task query; use sub-agent reply directly (skip travel aggregation)",
            ),
            "memory_aggregation_instruction": (
                "__const__",
                "Answer only within the scope of the user's original request. "
                "Do not add itinerary/hotel/food content unless explicitly requested.",
            ),
        },
    },
}


def _resolve(module_name: str, attr: str) -> str:
    if module_name == "__const__":
        return attr
    import importlib

    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def export_domain(domain: str) -> None:
    mapping = DOMAIN_FIELD_MAP[domain]
    root = ROOT / "domains" / domain / "locales"
    root.mkdir(parents=True, exist_ok=True)
    for locale, fields in mapping.items():
        payload = {key: _resolve(mod, attr) for key, (mod, attr) in fields.items()}
        out = root / f"{locale}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")


def main() -> None:
    for domain in ("travel",):
        export_domain(domain)


if __name__ == "__main__":
    main()
