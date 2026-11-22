"""Orchestration for the travel multi-agent system.

Chapter-8 ships a single orchestration strategy: the LangGraph fixed graph.
    fixed_graph/            - decompose -> route -> layered A2A execution -> aggregate
    pipeline.py             - PipelineConfig toggles
    aggregation_helpers.py  - single-task direct reply vs multi-task composition"""
