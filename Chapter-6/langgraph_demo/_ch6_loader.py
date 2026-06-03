"""从 Chapter-6 父目录加载模块，避免 Chapter-6/langgraph_demo 目录名遮蔽 pip langgraph_demo 包"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

CHAPTER6_DIR = Path(__file__).resolve().parent.parent
BOOK_ROOT = CHAPTER6_DIR.parent

# 这些模块会间接 import langchain → langgraph_demo，加载时不能让 Chapter-6 在 sys.path 中
_LANGCHAIN_DEPENDENT = frozenset({"sub_agents"})


def load_ch6_module(name: str) -> ModuleType:
    """加载 Chapter-6/*.py 模块"""
    key = f"ch6.{name}"
    if key in sys.modules:
        return sys.modules[key]

    path = CHAPTER6_DIR / f"{name}.py"
    if not path.exists():
        raise ImportError(f"Chapter-6 模块不存在: {path}")

    saved_path = sys.path[:]

    if name in _LANGCHAIN_DEPENDENT:
        # 移除 Chapter-6，防止 `import langgraph_demo` 命中 Chapter-6/langgraph_demo/ 目录
        sys.path[:] = [
            p for p in sys.path
            if Path(p).resolve() != CHAPTER6_DIR.resolve()
        ]
    elif str(CHAPTER6_DIR) not in sys.path:
        sys.path.insert(0, str(CHAPTER6_DIR))

    try:
        spec = importlib.util.spec_from_file_location(key, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved_path


def import_pip_langgraph(submodule: str = "") -> ModuleType:
    """导入 site-packages 中的 langgraph_demo"""
    import importlib

    blocked = {str(CHAPTER6_DIR), str(CHAPTER6_DIR / "langgraph_demo")}
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if p not in blocked]
    try:
        name = f"langgraph_demo.{submodule}" if submodule else "langgraph_demo"
        return importlib.import_module(name)
    finally:
        sys.path[:] = saved
