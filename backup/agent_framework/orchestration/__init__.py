"""中心编排层（Orchestration）。

本包实现 LangGraph 固定图工作流，将领域无关的调度逻辑与具体 Agent 实现分离：

    orchestrator.py  — 对外入口 LangGraphOrchestrator
    fixed_graph/
        graph.py     — 按 PipelineConfig 组装 StateGraph 节点与边
        nodes.py     — 各图节点实现（预调查 / 记忆 / 规划 / 执行 / 聚合）
        state.py     — 图状态 TypedDict
        stream_sink.py — 流式 token / 进度回调桥
        visualize.py — 图结构导出（Mermaid / PNG）
"""
