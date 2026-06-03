"""Fixed_graph 包内测试入口。

推荐运行（在 Chapter-6 目录）：
    python -m fixed_graph.test

也支持 PyCharm 直接运行本文件。
"""

import asyncio
import sys
from pathlib import Path

_CH6 = Path(__file__).resolve().parent.parent
if str(_CH6) not in sys.path:
    sys.path.insert(0, str(_CH6))

from fixed_graph.orchestrator import LangGraphOrchestrator


async def main():
    """主函数：测试 LangGraph 中心智能体"""
    orchestrator = LangGraphOrchestrator(enable_memory=True)

    orchestrator.show_graph()
    orchestrator.save_graph()

    result = await orchestrator.process_request("查询上海2026年6月3号天气", thread_id="lg_002")
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
