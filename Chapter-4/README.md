# Chapter 4: Task Decomposition, Dependency Planning, and Replanning

Author: zhanghong, liuhui

This chapter uses capability-based decomposition and dependency ordering, then
adds conflict-driven replanning and an explicit infeasible branch. A
reader-facing itinerary layer turns the validated task plan into a concrete
three-day draft and applies runtime changes to that draft.

All teaching results are produced by live model calls. There is no fixture,
cached response, or preset model output.

## Files

| File | Purpose |
| --- | --- |
| `task_decomposition_and_planning.ipynb` | Main teaching notebook: decomposition, dependency ordering, initial plan, replanning, and infeasible handling. |
| Root `llm.py` | Shared model configuration, text invocation, and JSON invocation helpers. |
| `task_decomposition.py` | Reusable decomposition prompt, parser, and model call. |
| `dependency_analysis.py` | Reusable dependency prompt, graph validation, and execution ordering. |
| `conflict_replanning.py` | Incremental plan construction, conflict-driven replanning, validation, and infeasible handling. |
| `travel_plan_generation.py` | Live itinerary generation, constraint validation, compact teaching views, and concrete itinerary revision. |

## Environment

Python 3.10 or later is recommended. Install the dependencies from this
directory:

```bash
python -m pip install -r requirements.txt
```

Configure the existing OpenAI-compatible variables in the repository-root
`.env`, or set the same variables in the process environment:

```text
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.example.com
OPENAI_MODEL=your_model_name
```

`OPENAI_MODEL` may be any model exposed by your provider. To use Qwen via DashScope, configure `DASHSCOPE_API_KEY`, `DASHSCOPE_CHAT_BASE_URL`, and `DASHSCOPE_CHAT_MODEL` instead. This chapter reuses the
repository's shared model layer and root environment; it does not
maintain a chapter-local `llm.py` or `.env`. Never store a real API key in the
Notebook or commit it to source control.

## Run

Start Jupyter in this directory, open
`task_decomposition_and_planning.ipynb`, and run all cells from top to bottom.
The Notebook calls the configured model for task decomposition, dependency ordering, a
resolvable runtime conflict, and an infeasible runtime conflict. Missing model
credentials or invalid JSON responses are reported directly; the program does
not fall back to a preset answer.

## Relationship to later chapters

The Notebook remains capability-based. Chapter 6 can continue to reuse
`task_decomposition.py` and `dependency_analysis.py` when concrete domain
agents are introduced. The new `conflict_replanning.py` module operates on the
plan produced by those existing stages and does not replace their interfaces.

---

# 第四章：任务拆解、依赖规划与冲突重规划

作者：zhanghong、liuhui

本章从能力目录出发完成任务拆解与依赖排序，并进一步生成具体三日行程，处理运行时冲突和不可行情形。

全部教学结果均由真实模型调用生成，不提供fixture、缓存响应或预设模型输出。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `task_decomposition_and_planning.ipynb` | 教学主入口，依次展示任务拆解、依赖排序、初始计划、冲突重规划和不可行处理。 |
| 根目录`llm.py` | 全书共享的模型配置、文本调用和JSON调用封装。 |
| `task_decomposition.py` | 可复用的任务拆解提示词、响应解析和模型调用。 |
| `dependency_analysis.py` | 可复用的依赖分析、依赖图校验和执行排序。 |
| `conflict_replanning.py` | 计划构造、冲突重规划、结果校验和不可行处理。 |
| `travel_plan_generation.py` | 真实模型调用下的行程生成、约束校验、教学摘要和具体行程修订。 |

## 环境配置

建议使用Python 3.10或更高版本。在本目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

在代码仓库根目录的`.env`中配置OpenAI兼容变量，或者在进程环境中配置相同变量：

```text
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.example.com
OPENAI_MODEL=your_model_name
```

`OPENAI_MODEL`可配置为服务商提供的任意模型；使用通义千问（DashScope）时，改为配置 `DASHSCOPE_API_KEY`、`DASHSCOPE_CHAT_BASE_URL` 和 `DASHSCOPE_CHAT_MODEL`。本章直接复用代码仓库根目录的共享模型层和环境配置，不维护章节局部`llm.py`或`.env`。真实API密钥不得写入Notebook，也不得提交到代码仓库。

## 运行方式

在本目录启动Jupyter，打开`task_decomposition_and_planning.ipynb`，从上到下执行全部单元。Notebook会真实调用所配置的模型完成任务拆解、依赖排序、候选行程生成、可解决冲突重规划和不可解决冲突判断。缺少凭据或模型未返回合法JSON时，程序会直接报告错误，不会回退到预设答案。

## 教学边界

本章仍采用“基于能力”的规划方式，不提前引入具体领域智能体的创建、路由和通信。运行时冲突作为明确输入交给规划器；模型提出修订候选后，程序继续检查任务引用、依赖关系、拓扑顺序以及未受影响任务是否被擅自改写。若原目标和新冲突无法同时满足，程序返回结构化不可行结果并请求用户决定，而不是虚构可行计划。

第6章仍可继续复用`task_decomposition.py`和`dependency_analysis.py`。`conflict_replanning.py`只处理上述阶段生成的计划，不替换已有接口。
