import sys
from pathlib import Path
import asyncio

CHAPTER6 = Path.cwd()  # 若在 Chapter-6 目录
sys.path.insert(0, str(CHAPTER6))
sys.path.insert(0, str(CHAPTER6 / "langgraph_demo"))

from orchestrator import LangGraphOrchestrator


async def main():
    """主函数：测试 LangGraph 中心智能体"""
    orchestrator = LangGraphOrchestrator(enable_memory=True)

    # 查看 / 保存图结构
    orchestrator.show_graph()
    orchestrator.save_graph()

    result = await orchestrator.process_request("查询上海2026年6月1号天气", thread_id="lg_002")
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())