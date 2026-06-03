"""
存储与检索示例 - Memory Store & Retrieve（混合搜索）

演示如何将记忆存入 Chroma 向量库，并通过混合检索（语义+关键词）找回它。
"""

import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ---------------------------------------------------------------------------
# 1. 配置 Chroma 客户端与 Embedding 模型
# ---------------------------------------------------------------------------
api_key = os.getenv("DASHSCOPE_API_KEY")
model_name = "text-embedding-v3"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = chromadb.PersistentClient(path="./chroma_demo_temp")
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name=model_name,
    api_base=base_url,
)

# 获取或创建集合
collection = client.get_or_create_collection(
    name="simple_memory_demo",
    embedding_function=embed_fn,
)

# ---------------------------------------------------------------------------
# 2. 存储：写入几条示例记忆
# ---------------------------------------------------------------------------
print("=" * 70)
print("【步骤 1】存储：将记忆写入 Chroma 向量库")
print("-" * 70)

memories = [
    {"id": "m1", "text": "用户喜欢住海景房，预算每晚不超过800元。"},
    {"id": "m2", "text": "用户是素食主义者，不吃海鲜。"},
    {"id": "m3", "text": "用户计划下个月去三亚旅游。"}
]

for mem in memories:
    collection.add(ids=[mem["id"]], documents=[mem["text"]])
    print(f"  ✅ 已存入: {mem['text']}")

# ---------------------------------------------------------------------------
# 3. 检索：通过问题查找相关记忆
# ---------------------------------------------------------------------------
query = "你还记得我喜欢住什么样的酒店吗？"
print("\n" + "-" * 70)
print(f"【步骤 2】检索：针对问题「{query}」进行搜索")
print("-" * 70)

results = collection.query(
    query_texts=[query],
    n_results=2,
)

print("\n【模型找到的相关记忆】")
for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    # 距离越小，相似度越高
    similarity = 1.0 - (dist / 2.0) 
    print(f"  [{i+1}] {doc}")
    print(f"      相似度: {similarity:.4f}")

print("\n" + "=" * 70)
print("演示结束。这就是大模型拥有‘长期记忆’的基础原理。")
# ======================================================================
# 【步骤 1】存储：将记忆写入 Chroma 向量库
# ----------------------------------------------------------------------
#   ✅ 已存入: 用户喜欢住海景房，预算每晚不超过800元。
#   ✅ 已存入: 用户是素食主义者，不吃海鲜。
#   ✅ 已存入: 用户计划下个月去三亚旅游。
#
# ----------------------------------------------------------------------
# 【步骤 2】检索：针对问题「你还记得我喜欢住什么样的酒店吗？」进行搜索
# ----------------------------------------------------------------------
#
# 【模型找到的相关记忆】
#   [1] 用户喜欢住海景房，预算每晚不超过800元。
#       相似度: 0.5494
#   [2] 用户计划下个月去三亚旅游。
#       相似度: 0.4836
#
# ======================================================================
# 演示结束。这就是大模型拥有‘长期记忆’的基础原理。