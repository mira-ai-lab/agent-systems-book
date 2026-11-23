# 测试索引

默认运行（无需 live LLM）：

```bash
cd Chapter-11
python -m pytest -m "not integration" -q
```

集成测试（需 `DASHSCOPE_API_KEY` 等）：`python -m pytest -m integration`

## 模块 → 测试文件

| 模块 / 主题 | 主要测试文件 |
|-------------|--------------|
| **Router Engine** | `test_phase10_router.py`, `test_phase11_router.py`, `test_phase14_extraction.py`, `test_phase27_travel_router.py` |
| **Profile / 自动路由** | `test_phase16_product.py`, `test_phase24_p2.py` |
| **编排 Fixed Graph** | `test_orchestration_graph.py`, `test_orchestration_orchestrator.py`, `test_run_demo_legacy_graph.py` |
| **Supervisor / handoff** | `test_phase7_supervisor.py`, `test_phase29_subtask_stream.py` |
| **A2A** | `test_phase7_a2a.py`, `test_phase7d_a2a_server.py` |
| **领域插件** | `test_platform_plugin.py`, `test_domain_registry.py`, `test_domain_parsing.py` |
| **TaskPlanner** | `test_planner.py`, `test_domain_task_planner.py`, `test_planner_pipeline.py` |
| **Knowledge Base** | `test_phase15_product.py`, `test_phase21_product.py`, `test_phase24_product.py`, `test_phase25_kb_*.py` |
| **Dynamic Registry** | `test_phase12_dynamic_registry.py`, `test_phase24_registry.py`, `test_phase25_registry_federation.py` |
| **i18n / locales** | `test_phase24_i18n.py`, `test_phase18_product.py` |
| **HTTP API / Jobs** | `test_api.py`, `test_phase26_jobs.py`, `test_phase20_product.py` |
| **Observability** | `test_tracing.py`, `test_trace_provider.py`, `test_phase8_observability.py`, `test_phase8b_metrics.py` |
| **Travel benchmark** | `test_travel_*benchmark*.py`, `test_travel_single_agents.py`, `test_e2e_*.py` |
| **TextGrad / optimization** | `test_textgrad_*.py`, `test_optimization_core_textgrad.py`, `test_agent_pipeline.py`, `test_mini_pipeline.py` |
| **Router Client SDK** | `test_phase25_router_client.py`, `test_phase26_router_client_integration.py`, `test_phase26_demo_web.py` |
| **产品就绪度** | `test_phase24_p2.py`（`product_readiness_check` 脚本） |
| **Phase 编号历史** | `test_phase{N}_*.py` — 对应 [UPGRADE.md](../UPGRADE.md) 各 Phase 交付项 |

## 命名说明

`test_phase*` 文件名保留升级阶段编号，便于与 UPGRADE 对照；新用例优先放入语义化文件名（如 `test_travel_routing_benchmark.py`）。
