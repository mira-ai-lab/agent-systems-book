"""Router 产品入口最小示例（推荐：Router → workflow / auto → 执行后端）。"""

from __future__ import annotations

import asyncio

from agent_framework.bootstrap import route
from agent_framework.config import load_project_dotenv


async def main() -> None:
    load_project_dotenv()
    result = await route(
        "给我查下北京到上海明天的机票",
        domain="travel",
        profile="workflow",
    )
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
