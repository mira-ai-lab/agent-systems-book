"""
Chapter-6 共享库：工具、Prompt、记忆、子智能体、中心编排。

子包：
  supervisor/   — Supervisor 动态 handoff + 规划流水线
  fixed_graph/  — LangGraph 固定 StateGraph
  supervisor/book/ — 书籍伪代码案例（不参与运行时）
"""

from chapter6.paths import BOOK_ROOT, CH6_DIR, CHROMA_DIR, load_project_dotenv

__all__ = [
    "BOOK_ROOT",
    "CH6_DIR",
    "CHROMA_DIR",
    "load_project_dotenv",
]
