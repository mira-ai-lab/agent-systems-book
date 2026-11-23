# Prompt 进化（Parallel Optimization Track）

本目录下的 `agent_framework/optimization/` 与生产 LangGraph **并行**：不改主编排代码，通过 benchmark 优化 prompt，产物写入 `data/benchmark/**/optimized/`。

## 安装

```powershell
cd Chapter-11
pip install -e ".[evolution]"   # 含 textgrad>=0.1.5
```

配置 `.env`：`DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY`），可选 `EXECUTOR_MODEL` / `OPTIMIZER_MODEL`。

## 优化后端

**Planner**（`optimize_travel_planner.py`）可选 `--backend`：

| backend | 依赖 textgrad | 说明 |
|---------|---------------|------|
| `local` | 否 | 失败反馈 + LLM 改写 prompt |
| `textgrad_lib` | 是 | `TextualGradientDescent`，失败文本作 loss |
| `textgrad_graph` | 是 | 计算图 `StringBasedFunction` + `MultiFieldEvaluation` |

**子 Agent**（`optimize_travel_agent.py`）**没有** `--backend` 参数：脚本固定走 **`textgrad_agent_graph`**（单 Agent 计算图 + TextGrad），需 `pip install -e ".[evolution]"`。

## Planner 目标（`--objective`）

| objective | 评测 / rollback | TextGrad forward |
|-----------|-----------------|------------------|
| `l1_l2`（默认） | L1 拆解 或 L2 路由规则分 | TaskPlanner 三步图 |
| `e2e` | `score_e2e_run` 规则分 | 完整编排（Router/LangGraph → Agent → 聚合） |

### E2E 规则分（`score_e2e_run`）

权重合计 1.0：

| 项 | 权重 | 规则 |
|----|------|------|
| 有回复 | 0.10 | `final_response` 非空 |
| 关键词 | 0.25（无 tool_checks）或 0.15（有 tool_checks） | 在 **final_response + agent_summary** 中匹配 |
| **tool_data** | 0.10（仅当配置 `tool_checks`） | 子任务 `tool_data` 字段子串，如 `city: 北京` |
| 禁用词 | 0.15 | 禁止词不得出现在上述语料中 |
| Agent | 0.35 | `subtask_results[*].agent` 覆盖期望列表 |
| 完成数 | 0.15 | `status` 为 `completed`/`ok` 的子任务数 ≥ 期望 |

规格与 TextGrad loss 标签对齐：见 `agent_framework/optimization/e2e/rules.py`。

### fixture 中配置 tool_checks（可选）

```json
"expect": {
  "tool_checks": [
    {
      "task_id": "T1",
      "field_contains": { "city": ["北京"] },
      "forbid_error": true
    }
  ]
}
```

支持 `tool_data.calls[]` 多工具输出；字段值做子串匹配（与关键词规则一致）。

### 规则分 vs TextGrad loss

- **rollback / 失败筛选**：始终用 `score_e2e_run`（确定性）。
- **改 prompt**：`MultiFieldEvaluation` 用 optimizer LLM 比较「实际 E2E JSON」与 `build_e2e_expectation_label`。
- 训练失败 case 会把 `score.details` 注入标签的 `rule_scorer_failures_on_this_run`，与 rollback 标准一致。

## 常用命令

```powershell
# L1/L2 + 计算图（教程 Notebook 同款）
python scripts/travel/optimize_travel_planner.py --backend textgrad_graph --objective l1_l2 --max-steps 3

# 完整 E2E 优化
python scripts/travel/optimize_travel_planner.py --backend textgrad_graph --objective e2e --e2e-timeout 300 --max-failure-cases 3

# E2E 评测（不调 TextGrad）
python scripts/travel/eval_travel_e2e.py --split dev

# 单 Agent system_prompt 优化（固定 textgrad_agent_graph，无需 --backend）
python scripts/travel/optimize_travel_agent.py --agent FlightAgent --max-steps 3

# 多 Agent 并列/顺序优化
python scripts/travel/optimize_travel_agent.py --agent all --max-steps 3
python scripts/travel/optimize_travel_agent.py --agent all --sequential --max-steps 3

# mini-pipeline 串联优化（见 optimize_travel_agent_pipeline.py）
python scripts/travel/optimize_travel_agent_pipeline.py --max-steps 3
```

## 成本提示（objective=e2e）

每步优化大致：

1. train：每个 case **一次** `process_request`（`evaluate_e2e_train_cases`）
2. backward：每个失败 case **再一次** E2E（TextGrad forward，默认最多 `--max-failure-cases 3`，按分数从低到高）
3. dev rollback：dev split 再跑一遍规则评测

建议 smoke：`--train-split dev --max-steps 1 --max-failure-cases 1`；正式：`train` split + `--e2e-timeout 300`。

## 产物

| 路径 | 内容 |
|------|------|
| `data/benchmark/travel_planner/optimized/zh.json` | `decomposition_prompt`, `agent_routing` |
| `data/benchmark/travel_planner/planner_*_optimization_report.json` | 逐步 ACCEPT/REJECT |
| `data/benchmark/travel_agents/optimized/zh.json` | 各 Agent `system_prompt` |

运行时 `TravelPrompts.build(use_optimized=True)` 自动加载。

## 教程

- L1/L2：`notebooks/planner_b1_textgrad_graph.ipynb`
- **完整 E2E**：`notebooks/planner_e2e_textgrad_graph.ipynb`

## 测试

```powershell
pytest tests/test_travel_e2e_benchmark.py tests/test_e2e_collect.py tests/test_textgrad_graph.py -q
pytest -m textgrad   # 需安装 textgrad，部分用例需 API
```
