# Chapter 4: Task Decomposition and Dependency Planning

Author: zhanghong, liuhui · Last updated: 2026-11-17

## Files

| File | Purpose |
| --- | --- |
| `task_decomposition_and_planning.ipynb` | Capability-based teaching notebook. It explains how a model decomposes a goal using available capabilities. |
| Root `llm.py` | Shared model configuration and invocation helpers. |
| `task_decomposition.py` | Reusable agent-based decomposition prompt, response parser, and model call. |
| `dependency_analysis.py` | Reusable dependency prompt, graph validation, and execution ordering. |

## Relationship to later chapters

The Notebook deliberately remains capability-based and does not import the two
reusable modules. Chapter 6 introduces the concrete domain-agent team and
imports `task_decomposition.py` and `dependency_analysis.py` to build its
centralized execution plan. Their prompts are synchronized from the Notebook;
the dependency module only extends the output with `depends_on` for routing.

---

# 第四章：任务拆解与依赖规划

作者：zhanghong, liuhui · 更新时间：2026-11-17

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `task_decomposition_and_planning.ipynb` | 基于能力的教学 Notebook，说明如何根据可用能力拆解用户目标。 |
| 根目录 `llm.py` | 统一的模型配置与调用封装。 |
| `task_decomposition.py` | 供后续章节复用的任务拆解提示词、响应解析和模型调用。 |
| `dependency_analysis.py` | 供后续章节复用的依赖分析、依赖图校验和执行排序。 |

## 与后续章节的关系

本章 Notebook 保持“基于能力”的教学方式，不引用这两个复用模块。第 6 章引入具体的领域智能体团队后，会从本章导入 `task_decomposition.py` 和 `dependency_analysis.py`，生成可执行的中心化任务计划。

两个模块的提示词以本章 Notebook 为准；依赖分析模块仅为第 6 章路由补充 `depends_on` 输出字段。
