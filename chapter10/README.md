# Chapter-10：提示词自进化（两种 TextGrad 模式 + rollback）

在第 8 章建立的固定图多智能体系统上演示提示词自进化：用 benchmark 评测 L1（任务拆解）
/ L2（智能体路由）基线 → 收集失败样例 → 用 TextGrad 生成文本梯度更新提示词 → 在 dev
上复评，只有不退化才接受（rollback）。

本章提供两种 TextGrad 用法，入口都是 `optimize.run_planner_optimization(backend=...)`：

| 模式 | backend 参数 | 机制 |
|------|------|------|
| direct | `"direct"` | 只把 prompt 当作可训练变量，loss 让 LLM 直接批评「prompt + 失败样例」，再 `TextualGradientDescent` 更新 |
| planner_graph | `"planner_graph"` | 把「拆解→依赖→路由」整条流水线用 `StringBasedFunction` 建成计算图，loss 比较「流水线输出 vs 期望」，文本梯度穿过真实步骤回传 |

## 使用的库：textgrad

| 项 | 内容 |
|----|------|
| 包名 | `textgrad` |
| 版本 | 0.1.8（MIT License，作者 Zou Group / Stanford） |
| GitHub | https://github.com/zou-group/textgrad |
| 作用 | TextGrad 用「通过文本求导」的方式做自动优化：把提示词等文本对象当作可微分变量，用语言模型计算自然语言的「文本梯度」，再以 `TextualGradientDescent` 迭代改进文本，直到评测指标不再提升。 |

安装（书仓库 venv）：

```powershell
.venv\Scripts\python.exe -m pip install "textgrad>=0.1.5"
```

> 注：textgrad 会带入 litellm 等依赖；本章只用 `Variable` / `TextLoss` /
> `MultiFieldEvaluation` / `TextualGradientDescent`，并通过自定义的 LangChain
> `EngineLM` 调模型，不直接调用 litellm。

## 目录

```
chapter10/
├── fixtures.json                  # travel 基准（15 个用例，train/dev/test 切分）
├── optimizable_prompts.py                     # 待优化的简单拆解/路由提示词模板 + 占位符 + 模板提取校验
├── runtime.py                     # Planner（拆解 / 依赖 / 路由，可注入候选 prompt）
├── evaluate.py                    # 规则化评测（L1 拆解 + L2 路由 + 参数完整性）
├── textgrad_engine.py             # LangChain ChatOpenAI -> textgrad EngineLM
├── planner_graph.py               # planner_graph 后端（同步桥 + 计算图 + 出候选）
├── optimize.py                    # direct 后端 + 共享优化循环 + 编排 / 保存
├── travel_prompt_evolution.ipynb  # 入口 Notebook（两种模式各跑一遍）
└── output/                        # 优化产物（按 backend 分目录）
```

## 运行

```powershell
cd Chapter-8
pip install -e .                    # 第8章重构后的包
cd ..\chapter10
jupyter notebook travel_prompt_evolution.ipynb
```

需要 `DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY`）。Notebook 默认用 `train=dev=dev`、
`max_steps=1` 快速跑通；正式实验改成 `train_split="train"`、`max_steps=3`。

## 说明

本目录使用 `optimizable_prompts.py` 中自带的简单拆解/路由提示词（不复用第 4/6 章模板），
仅复用第 4 章的拆解解析/依赖分析、第 6 章的领域定义与第 8 章的 `travel_multi_agent` 包，
以及仓库根 `llm.py` 与 `textgrad`；不依赖 `Chapter-11`，可独立运行。
