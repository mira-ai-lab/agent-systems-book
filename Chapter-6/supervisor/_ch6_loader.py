"""模块加载：优先 supervisor/ 本地，避免遮蔽 pip langgraph_demo 包"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SUP_DIR = Path(__file__).resolve().parent
BOOK_ROOT = SUP_DIR.parent
CH6_FALLBACK = BOOK_ROOT / "Chapter-6"

_LANGCHAIN_DEPENDENT = frozenset({"sub_agents"})


def _resolve_module_path(name: str) -> Path:
    local = SUP_DIR / f"{name}.py"
    if local.exists():
        return local
    fallback = CH6_FALLBACK / f"{name}.py"
    if fallback.exists():
        return fallback
    raise ImportError(f"模块不存在: {name}.py（已查找 {local} 与 {fallback}）")


def load_ch6_module(name: str) -> ModuleType:
    key = f"sup.{name}"
    if key in sys.modules:
        return sys.modules[key]

    path = _resolve_module_path(name)
    mod_dir = path.parent
    saved_path = sys.path[:]

    if name in _LANGCHAIN_DEPENDENT:
        sys.path[:] = [p for p in sys.path if Path(p).resolve() != SUP_DIR.resolve()]
        import_pip_langgraph()
        import_pip_langgraph("_internal")
    elif str(mod_dir) not in sys.path:
        sys.path.insert(0, str(mod_dir))

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
    import importlib

    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != SUP_DIR.resolve()]
    try:
        name = f"langgraph_demo.{submodule}" if submodule else "langgraph_demo"
        return importlib.import_module(name)
    finally:
        sys.path[:] = saved
