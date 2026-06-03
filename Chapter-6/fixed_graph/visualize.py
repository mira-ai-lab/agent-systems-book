"""LangGraph 工作流图结构可视化 — 全部由编译后的图对象自动生成"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import bootstrap

bootstrap.setup()

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PKG_DIR / "output"


def compile_graph_for_visualization() -> Any:
    """仅编译图结构，不执行节点（无需有效 API Key）"""
    from langchain_openai import ChatOpenAI

    from .graph import build_central_agent_graph

    llm = ChatOpenAI(
        model="qwen-plus",
        api_key="sk-graph-visualization-only",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return build_central_agent_graph(llm, None).compile()


class GraphVisualizer:
    """从 LangGraph 编译结果导出图结构（Mermaid / ASCII / PNG / 文本摘要）"""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._graph = app.get_graph()

    @classmethod
    def from_compiled(cls, app: Any) -> "GraphVisualizer":
        return cls(app)

    @classmethod
    def standalone(cls) -> "GraphVisualizer":
        return cls(compile_graph_for_visualization())

    def get_nodes(self) -> List[str]:
        return sorted(self._graph.nodes.keys())

    def get_edges(self) -> List[Any]:
        return list(self._graph.edges)

    def get_mermaid(self) -> str:
        """LangGraph 原生 Mermaid 输出"""
        return self._graph.draw_mermaid()

    def get_ascii(self) -> str:
        """LangGraph 原生 ASCII 输出（需要 grandalf；含自循环条件边时可能失败）"""
        return self._graph.draw_ascii()

    def _try_ascii(self) -> Optional[str]:
        try:
            return self.get_ascii()
        except ImportError as exc:
            return f"跳过 ASCII: {exc}\n  安装: pip install grandalf"
        except Exception as exc:
            return (
                f"ASCII 生成失败（LangGraph/grandalf 对自循环条件边支持有限）: {exc}\n"
                "  请使用 output/*.png 或 *.mmd 查看图结构"
            )

    def get_png_bytes(self) -> bytes:
        """LangGraph 原生 PNG 输出（内部走 Mermaid 渲染）"""
        return self._graph.draw_mermaid_png()

    def get_structure_text(self) -> str:
        """从图对象的 nodes / edges 自动生成的文本摘要"""
        lines = [
            "=" * 60,
            "LangGraph StateGraph — 自动导出的图结构",
            "=" * 60,
            "",
            f"节点数: {len(self.get_nodes())}",
            f"边数:   {len(self.get_edges())}",
            "",
            "【节点】",
        ]
        for name in self.get_nodes():
            node = self._graph.nodes[name]
            lines.append(f"  • {name}  ({type(node.data).__name__ if node.data else 'terminal'})")

        lines.extend(["", "【边】"])
        for edge in self.get_edges():
            kind = "条件边" if edge.conditional else "固定边"
            lines.append(f"  • [{kind}] {edge.source} → {edge.target}")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def print_all(self) -> None:
        print(self.get_structure_text(), flush=True)

        print("\n--- Mermaid（LangGraph 生成，可粘贴到 https://mermaid.live）---", flush=True)
        print(self.get_mermaid(), flush=True)

        print("\n--- ASCII（LangGraph 生成）---", flush=True)
        ascii_text = self._try_ascii()
        if ascii_text:
            print(ascii_text, flush=True)

    def save_all(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "central_agent_graph",
    ) -> Dict[str, Path]:
        out = Path(output_dir or DEFAULT_OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        mermaid_path = out / f"{prefix}.mmd"
        mermaid_path.write_text(self.get_mermaid(), encoding="utf-8")
        paths["mermaid"] = mermaid_path

        summary_path = out / f"{prefix}.txt"
        summary = self.get_structure_text()
        ascii_text = self._try_ascii()
        if ascii_text:
            summary += "\n\n--- ASCII (LangGraph) ---\n" + ascii_text
        summary_path.write_text(summary, encoding="utf-8")
        paths["text"] = summary_path

        png_path = out / f"{prefix}.png"
        png_path.write_bytes(self.get_png_bytes())
        paths["png"] = png_path

        print(f"✓ 图结构已保存到 {out.resolve()}", flush=True)
        for kind, path in paths.items():
            print(f"  - {kind}: {path.name}", flush=True)
        return paths
