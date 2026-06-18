"""run_demo.py --legacy-graph CLI 开关。"""

from pathlib import Path

RUN_DEMO = Path(__file__).resolve().parent.parent / "scripts" / "run_demo.py"


def test_run_demo_supports_legacy_graph_flag():
    src = RUN_DEMO.read_text(encoding="utf-8")
    assert "--legacy-graph" in src
    assert "LangGraphOrchestrator" in src
    assert "_create_legacy_orchestrator" in src
    assert "run_legacy_chat" in src
