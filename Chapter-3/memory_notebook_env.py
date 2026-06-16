"""Chapter-3 记忆相关 notebook 的公共环境加载工具。"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class NotebookEnv:
    """notebook 运行上下文。"""

    chapter3: Path
    root: Path

    @property
    def detail_notebook(self) -> Path:
        return self.chapter3 / "memory_text_vector_detail.ipynb"


def configure_stdio_utf8() -> None:
    """Windows 下避免 print 中文乱码。"""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def resolve_chapter3_paths(start: Path | None = None) -> NotebookEnv:
    """解析 Chapter-3 与项目根目录（兼容从 Chapter-3/ 或仓库根打开 notebook）。"""
    start = start or Path.cwd()

    if (start / "memory_text_vector_detail.ipynb").exists():
        chapter3 = start
    elif (start / "Chapter-3" / "memory_text_vector_detail.ipynb").exists():
        chapter3 = start / "Chapter-3"
    else:
        chapter3 = start

    root = chapter3.parent if (chapter3.parent / ".env").exists() else chapter3
    return NotebookEnv(chapter3=chapter3.resolve(), root=root.resolve())


def setup_notebook_env(start: Path | None = None, *, verbose: bool = True) -> NotebookEnv:
    """一键初始化 notebook 环境：UTF-8、加载 .env、返回路径。"""
    configure_stdio_utf8()
    env = resolve_chapter3_paths(start)
    load_dotenv(env.root / ".env")
    load_dotenv(env.chapter3 / ".env")

    if verbose:
        print("Chapter-3:", env.chapter3)
        print("项目根目录:", env.root)
        print(
            "DASHSCOPE_API_KEY:",
            "已配置" if os.getenv("DASHSCOPE_API_KEY", "").strip() else "未配置",
        )
    return env


def import_memory_defs(
    namespace: dict[str, Any] | None = None,
    *,
    env: NotebookEnv | None = None,
    verbose: bool = True,
) -> int:
    """从 .py 模块导入源码 A/B 定义，注入到 namespace。"""
    env = env or resolve_chapter3_paths()
    chapter3 = env.chapter3
    if str(chapter3) not in sys.path:
        sys.path.insert(0, str(chapter3))

    from memory_hybrid_compressor import HybridRetriever, MemoryCompressor
    from memory_ltm_improved import (
        MEMORY_PROMPT_TEMPLATE,
        MemoryWritePipeline,
        SelfBuiltLongTermMemoryImproved,
        ThreadShortTermMemory,
        WritePipelineConfig,
        close_chroma_ltm,
        default_llm,
        get_or_reset_ltm,
        reset_chroma_directory,
    )

    ns = namespace if namespace is not None else globals()
    ns.setdefault("ROOT", env.root)
    ns.setdefault("CHAPTER3", env.chapter3)
    exports = {
        "HybridRetriever": HybridRetriever,
        "MemoryCompressor": MemoryCompressor,
        "SelfBuiltLongTermMemoryImproved": SelfBuiltLongTermMemoryImproved,
        "MemoryWritePipeline": MemoryWritePipeline,
        "ThreadShortTermMemory": ThreadShortTermMemory,
        "WritePipelineConfig": WritePipelineConfig,
        "MEMORY_PROMPT_TEMPLATE": MEMORY_PROMPT_TEMPLATE,
        "close_chroma_ltm": close_chroma_ltm,
        "default_llm": default_llm,
        "get_or_reset_ltm": get_or_reset_ltm,
        "reset_chroma_directory": reset_chroma_directory,
    }
    ns.update(exports)

    if verbose:
        print("✓ 已从 memory_hybrid_compressor / memory_ltm_improved 加载定义")
    return len(exports)


def load_detail_notebook_defs(
    namespace: dict[str, Any] | None = None,
    *,
    env: NotebookEnv | None = None,
    markers: tuple[str, ...] = ("源码 A", "源码 B"),
    verbose: bool = True,
    use_modules: bool = True,
) -> int:
    """加载源码 A/B：默认 import .py 模块；use_modules=False 时回退 exec notebook 代码格。"""
    if use_modules:
        return import_memory_defs(namespace, env=env, verbose=verbose)

    env = env or resolve_chapter3_paths()
    nb_path = env.detail_notebook
    if not nb_path.exists():
        raise FileNotFoundError(f"未找到 {nb_path}")

    ns = namespace if namespace is not None else globals()
    ns.setdefault("ROOT", env.root)
    ns.setdefault("CHAPTER3", env.chapter3)

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    loaded = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if any(marker in src for marker in markers):
            exec(src, ns)
            loaded += 1

    if verbose:
        print(f"✓ 已从 {nb_path.name} 加载 {loaded} 个源码格")
    return loaded
