"""直连 LangGraphOrchestrator 的最小示例（legacy 书稿路径，非 Router 产品入口）。"""

from __future__ import annotations

import asyncio

from agent_framework.config import load_project_dotenv
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.fixed_graph import LangGraphOrchestrator
from domains.travel import TravelPrompts, create_travel_registry, travel_domain_config


async def main() -> None:
    load_project_dotenv()
    orchestrator = LangGraphOrchestrator(
        registry=create_travel_registry(),
        prompts=TravelPrompts.build(),
        domain_config=travel_domain_config(enable_guess_agent=True),
        pipeline=PipelineConfig(enable_pre_survey=True, enable_memory=True),
    )
    result = await orchestrator.process_request("查询上海明天天气")
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
