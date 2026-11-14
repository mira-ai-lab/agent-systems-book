# Chapter 3：用户画像与长期记忆

Author: zhanghong, liuhui  
Last updated: 2026-11-14

本章通过“记忆抽取 → 相似记忆合并 → 持久化存储 → 混合检索 → 上下文使用”的完整链路，演示如何从多轮对话中构建可复用的用户画像与偏好记忆。`memory.ipynb` 负责串联各模块并展示执行结果，具体业务能力拆分在以下 Python 文件中。

## Python 文件说明

| 文件 | 作用 |
| --- | --- |
| `llm.py` | 模型客户端和结构化输出辅助模块。`create_llm` 根据仓库根目录 `.env` 创建 OpenAI 或阿里云百炼兼容模型，Qwen 思考模式默认关闭；`chat` 发起 JSON 输出请求，`extract_response_text` 和 `parse_json_response` 负责提取与解析响应。 |
| `memory_extract.py` | 记忆抽取模块。调用大模型将非结构化对话转换为包含摘要、关键点、重要性和时间戳的结构化记忆；模型或 JSON 解析错误会直接向上抛出。 |
| `memory_merge.py` | 记忆合并模块。将语义或内容相近的多条记忆合并为一条综合记忆，保留重要信息并减少重复存储；模型不可用时提供基于首条记忆的降级结果。 |
| `memory_store.py` | 记忆存储模块。管理 Chroma 持久化集合、嵌入函数、相似去重、合并更新、后台异步写入以及 Notebook 会话内的存储实例复用与清理。 |
| `memory_retrieve.py` | 记忆检索模块。基于向量相似度、BM25 和 Reciprocal Rank Fusion（RRF）召回候选，并结合时间衰减和重要性计算最终排序。 |
| `memory_use.py` | 记忆使用模块。维护按线程隔离的短期对话，将短期上下文与检索到的长期记忆组合为提示词；Notebook 随后直接调用 `llm.invoke` 并提取回答文本。 |

## 模块关系

```text
memory_extract.py
        │
        ▼
memory_merge.py ──► memory_store.py ──► memory_retrieve.py ──► memory_use.py
                                                   ▲
                                                   │
                                               llm.py
```

`llm.py` 为抽取、合并和使用阶段提供统一的模型调用能力；`memory_store.py` 在写入重复记忆时调用 `memory_merge.py`，`memory_retrieve.py` 从存储中读取候选并排序，最终由 `memory_use.py` 组装模型上下文。

## 运行方式

在仓库根目录执行：

```powershell
uv pip install -r Chapter-3/requirements.txt
```

随后打开 `Chapter-3/memory.ipynb`，选择仓库根目录 `.venv` 对应的 Python 内核，并按照 Notebook 中的顺序运行。模型配置统一从仓库根目录 `.env` 加载，至少需要配置 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`。
