# 轨迹转换与失败归因使用指南

本文档说明如何将 **OpenTelemetry span JSONL**（如 `latc.travel-multi-agent` 系统导出的轨迹）转换为 **Who&When 格式**，再使用本仓库 `Automated_FA` 进行失败归因（AutoFA）。

---

## 整体流程

```
spans_*.jsonl          convert_spans_to_whowhen.py          Who&When/*.json
(你的系统轨迹)    ──►   (格式转换)                    ──►   (归因输入)
                                                              │
                                                              ▼
                                                    Automated_FA/inference.py
                                                              │
                                                              ▼
                                                    outputs/*.txt
                                                    (Agent / Step / Reason)
                                                              │
                                                              ▼ (可选)
                                                    Automated_FA/evaluate.py
                                                    (与人工标注对比准确率)
```

---

## 一、前置条件

1. 已安装项目依赖（见根目录 `requirements.txt`；**通义千问 API 模式只需 `openai`、`python-dotenv`**，不必能正常 import torch）
2. 轨迹文件为 **JSONL**：每行一个 span 对象
3. 当前转换脚本适配 span 命名规则（后缀匹配，不写死完整前缀）：
   - `*.planner.*`（排除 `.build_plan` 包装 span）
   - `*.agent.invoke`
   - `*.orchestration.aggregate`

---

## 二、配置环境变量

在 `Automated_FA/.env` 中填写（可参考 `.env.example`）：

```env
DASHSCOPE_API_KEY=sk-你的DashScope密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus

# 失败归因时注入 prompt 的「今天」（YYYY-MM-DD，见 2.1 节）
FA_REFERENCE_DATE=2026-06-15

INFERENCE_DEVICE=cuda:0
```

Key 获取：[DashScope 控制台](https://dashscope.console.aliyun.com/)

> **注意：** `.env` 已在 `.gitignore` 中，请勿提交到 Git。

使用通义千问 API 时**不需要**配置 `AZURE_OPENAI_*`（那是 GPT 模型用的）。

### 2.1 配置参考日期 `FA_REFERENCE_DATE`

失败归因（`step_by_step` 等）时，评估模型需要知道**「今天」是哪一天**，才能正确理解用户说的「下周」「未来2周」等相对时间。  
`FA_REFERENCE_DATE` 会在每一步评估 prompt 中注入，例如：

```
【评估专用·当前日期】
- 今天：2026-06-15
- 判断「下周」「未来2周」等相对时间时，必须以今天为基准推算……
```

**如何设置**

1. 编辑 `Chapter-9/Automated_FA/.env`，增加一行：

   ```env
   FA_REFERENCE_DATE=2026-06-15
   ```

2. **修改后必须重新运行 Step 2（`inference.py` 或 notebook 中的归因单元格）**，新日期才会进入 prompt。

3. 自检（在 `Automated_FA` 目录下）：

   ```powershell
   cd Chapter-9/Automated_FA
   python -c "from dotenv import load_dotenv; load_dotenv('.env'); from Lib.time_context import format_time_context_for_eval; print(format_time_context_for_eval())"
   ```

   输出中应包含 `- 今天：2026-06-15`。

**样例轨迹建议值**

| 样例文件 | 建议 `FA_REFERENCE_DATE` | 说明 |
|----------|--------------------------|------|
| `spans_20260615_*.jsonl` | `2026-06-15` | 与轨迹内系统时间锚点一致（「今天」= 2026-06-15，「下周」= 06-22～06-28） |

**优先级**

1. 代码显式传入 `reference_date=...`（一般不用）
2. 环境变量 `FA_REFERENCE_DATE`
3. 未配置时：使用**运行程序当天的真实日期**（`date.today()`）

**不配置会怎样？**

若你在 2026 年以外的日期运行归因，模型可能把轨迹里的 `2026-06-15` 误判为「遥远的未来」，从而在 Planner 阶段就报错，扫不到真正的根因（如 WeatherAgent）。  
复现 README 第五节实测结果时，建议固定为 `2026-06-15`。

---

## 三、轨迹转换

### 3.1 脚本位置

```
testdata/convert_spans_to_whowhen.py
```

### 3.2 基本用法

在项目根目录执行：

```powershell
cd D:\myproject\Agents_Failure_Attribution

python testdata/convert_spans_to_whowhen.py testdata/spans_20260615_170343.jsonl testdata/spans_20260615_180713.jsonl -o testdata/converted
```

默认输出到 `testdata/converted/`，输出文件名与输入 stem 一致：

| 输入 | 输出 |
|------|------|
| `spans_20260615_170343.jsonl` | `spans_20260615_170343.json` |
| `spans_20260615_180713.jsonl` | `spans_20260615_180713.json` |

### 3.3 常用参数

| 参数 | 说明 |
|------|------|
| `jsonl_paths` | 一个或多个 JSONL 文件（支持空格分隔传入多个） |
| `-o`, `--output-dir` | 输出目录，默认 `testdata/converted` |
| `--output-name` | 指定输出 JSON 文件名（**仅单文件时可用**） |
| `--ground-truth` | 期望的正确结果描述（建议填写） |
| `--mistake-agent` / `--mistake-step` | 人工标注，供 `evaluate.py` 使用 |

### 3.4 转换规则

**history 步骤顺序：**

1. **Planner**：所有 `.planner.*` span，按 `start_time` 排序
2. **Agent**：所有 `.agent.invoke` span，按 `execution.order` → `layer.tasks` → `task.id` 排序
3. **汇总**：`.orchestration.aggregate`，附 `final_response`

**agent 命名：**

- Agent 步骤 → `attributes.task.agent`（如 `WeatherAgent`）
- Planner 步骤 → `Planner.{步骤名}`
- 汇总 → `Orchestrator`

---

## 四、失败归因（Inference）

### 4.1 进入归因目录并运行

```powershell
cd D:\myproject\Agents_Failure_Attribution\Automated_FA

python inference.py `
  --method step_by_step `
  --model qwen-plus `
  --is_handcrafted False `
  --directory_path ../testdata/converted
```

API Key 从 `Automated_FA/.env` 自动读取，**无需**在命令行传 `--api_key`。

### 4.2 支持的模型

| 类型 | `--model` 取值 | 是否需要 Key |
|------|----------------|-------------|
| 通义千问 API | `qwen-plus`, `qwen-max`, `qwen-turbo`, `qwen-long` | 需要 `DASHSCOPE_API_KEY` |
| Azure GPT | `gpt-4o`, `gpt4`, `gpt4o-mini` | 需要 `AZURE_OPENAI_*` |
| 本地 HuggingFace | `qwen-7b`, `qwen-72b`, `llama-8b`, `llama-70b` | 不需要 Key，需要 GPU |

### 4.3 三种归因方法

| 方法 | 说明 |
|------|------|
| `all_at_once` | 一次性看完整 history，最快 |
| `step_by_step` | 逐步扫描，遇到第一个「有错」的步骤即停止（**推荐**） |
| `binary_search` | 二分定位 step，步骤多时更省 token |

### 4.4 查看输出

```powershell
type outputs\step_by_step_qwen-plus_alg_generated.txt
```

输出文件名格式：`{method}_{model}_alg_generated.txt`

**Step 编号** = `history` 数组下标，从 **0** 开始。

---

## 五、手动验证流程（推荐按此顺序跑）

以下命令已在仓库内 **实测通过**（2026-06-16），你可逐步复制执行验证。

### Step 0：确认 `.env` 已配置

```powershell
# 确认文件存在
dir D:\myproject\Agents_Failure_Attribution\Automated_FA\.env
```

`.env` 中至少要有 `DASHSCOPE_API_KEY=sk-...`。

### Step 1：转换两条样例轨迹

```powershell
cd D:\myproject\Agents_Failure_Attribution

python testdata/convert_spans_to_whowhen.py `
  testdata/spans_20260615_170343.jsonl `
  testdata/spans_20260615_180713.jsonl `
  -o testdata/converted
```

**预期终端输出（摘要）：**

```
# 轨迹 1 — 大同天气（单 agent）
Wrote testdata\converted\spans_20260615_170343.json
execution_order: ['T1']
history steps: 6
  Step 0: Planner.pre_survey
  Step 1: Planner.decomposition
  Step 2: Planner.dependency
  Step 3: Planner.routing
  Step 4: WeatherAgent
  Step 5: Orchestrator

# 轨迹 2 — 多城市旅行（4 agent）
Wrote testdata\converted\spans_20260615_180713.json
execution_order: ['T1', 'T2', 'T3', 'T4']
history steps: 9
  Step 0: Planner.pre_survey
  ...
  Step 4: WeatherAgent
  Step 5: HotelAgent
  Step 6: RestaurantAgent
  Step 7: ItineraryAgent
  Step 8: Orchestrator
```

**自检：**

```powershell
dir testdata\converted\spans_*.json
# 应看到 2 个文件，不能只有一个 20260615.json
```

### Step 2：运行归因

```powershell
cd Automated_FA

python inference.py `
  --method step_by_step `
  --model qwen-plus `
  --is_handcrafted False `
  --directory_path ../testdata/converted
```

**预期终端输出（摘要）：**

```
Successfully initialized DashScope client: https://dashscope.aliyuncs.com/compatible-mode/v1
Model: qwen-plus
Output will be saved to: outputs\step_by_step_qwen-plus_alg_generated.txt
Analysis finished. Output saved to outputs\step_by_step_qwen-plus_alg_generated.txt
```

耗时约 **30–60 秒**（2 条轨迹，step_by_step 模式，取决于 API 响应速度）。

### Step 3：查看归因结果

```powershell
type outputs\step_by_step_qwen-plus_alg_generated.txt
```

**预期关键结果（实测）：**

| 轨迹文件 | 预测 Agent | 预测 Step | 说明 |
|----------|-----------|-----------|------|
| `spans_20260615_170343.json` | `Planner.dependency` | 2 | LLM 在 Step 2 判定有错并停止（见下方注意事项） |
| `spans_20260615_180713.json` | `WeatherAgent` | 4 | 与人工分析一致：天气日期范围错误 |

### Step 4（可选）：评估准确率

若你有人工标注，可在转换时写入，再跑 evaluate：

```powershell
cd D:\myproject\Agents_Failure_Attribution

python testdata/convert_spans_to_whowhen.py testdata/spans_20260615_180713.jsonl `
  --mistake-agent WeatherAgent --mistake-step 4 -o testdata/converted

cd Automated_FA
python evaluate.py `
  --data_path ../testdata/converted `
  --eval_file outputs/step_by_step_qwen-plus_alg_generated.txt
```

---

## 六、已验证样例数据说明

仓库内两条测试轨迹：

| 文件 | 用户问题 | Agent 数量 | 主要问题（人工判断） |
|------|----------|-----------|---------------------|
| `spans_20260615_170343.jsonl` | 大同未来2周的天气如何 | T1（WeatherAgent） | 只返回 14 天预报，缺少完整 2 周 |
| `spans_20260615_180713.jsonl` | 上海/苏州/杭州多城市旅行规划 | T1–T4 | WeatherAgent 查了本周天气而非「下周」 |

---

## 七、实测中发现的问题与修复

| 问题 | 现象 | 处理 |
|------|------|------|
| 输出文件名冲突 | 两个 JSONL 都生成 `20260615.json`，后者覆盖前者 | 已改为 `{输入stem}.json` |
| 任务数写死 T1–T4 | 单 agent 轨迹排序异常 | 已从 span 动态推断 `execution.order` |
| 启动时 import torch 失败 | `numpy.ndarray` 报错，API 模式无法运行 | 已改为 API 模式不加载 torch |
| `.env` 不生效 | 原先 argparse 默认值为空格，未读环境变量 | 已改为从 `DASHSCOPE_API_KEY` 读取 |

---

## 八、常见问题

### Q1：`--is_handcrafted` 为什么要 False？

转换后的 JSON 用 `history[i].name` 标识 agent（如 `WeatherAgent`），必须设为 `False`。设为 `True` 时会读 `role` 字段，格式不匹配。

### Q2：step_by_step 为什么没在 WeatherAgent 停就报了 Planner 的错？

`step_by_step` 遇到**第一个** LLM 判为「有错」的步骤就停止。轨迹 1 中 LLM 在 Step 2（Planner.dependency）就给了 Yes，不会继续扫到 Step 4 的 WeatherAgent。对复杂轨迹可改用 `binary_search` 或 `all_at_once`。

### Q3：转换后 history 为空？

检查 span 名是否含 `.agent.invoke` / `.planner.` / `.orchestration.aggregate`。若命名不同，需调整 `convert_spans_to_whowhen.py` 中的后缀匹配规则。

### Q4：归因结果说「没有明显错误」？

- 检查 `ground_truth` 是否写的是**期望结果**而非系统实际输出
- 用 `--ground-truth` 传入更具体的正确标准

### Q5：PowerShell 里 tqdm 进度条显示乱码？

不影响运行，只要最后出现 `Analysis finished. Output saved to ...` 即成功。

### Q6：`FA_REFERENCE_DATE` 是什么？为什么要设成 `2026-06-15`？

归因 prompt 需要明确的「今天」才能判断日期是否合理。不设则用运行当天的日期；跑 2026-06-15 录制的样例时，建议在 `.env` 里写 `FA_REFERENCE_DATE=2026-06-15`，并**重新跑 inference**。详见第二节 2.1。

---

## 九、相关文件

| 文件 | 作用 |
|------|------|
| `testdata/spans_20260615_170343.jsonl` | 样例轨迹 1（大同天气） |
| `testdata/spans_20260615_180713.jsonl` | 样例轨迹 2（多城市旅行） |
| `testdata/convert_spans_to_whowhen.py` | span JSONL → Who&When JSON |
| `testdata/converted/*.json` | 转换输出（归因输入） |
| `Automated_FA/.env` | DashScope API Key、`FA_REFERENCE_DATE` 等配置 |
| `Automated_FA/.env.example` | 配置模板 |
| `Automated_FA/Lib/time_context.py` | 参考日期注入（`FA_REFERENCE_DATE`） |
| `Automated_FA/inference.py` | 归因入口 |
| `Automated_FA/outputs/step_by_step_qwen-plus_alg_generated.txt` | 归因输出（实测生成） |
| `Automated_FA/evaluate.py` | 准确率评估 |
| `Automated_FA/Lib/utils.py` | 三种 AutoFA 方法实现 |
