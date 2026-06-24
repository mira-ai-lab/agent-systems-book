"""候选 prompt 的 rollback（回滚）判定逻辑。

local 与 textgrad_lib 共用同一规则：dev 得分不升则拒绝候选，保留历史最优 prompt。
"""


def should_accept_candidate(
    candidate_dev_score: float,
    best_dev_score: float,
    *,
    rollback: bool = True,
) -> bool:
    """判断候选 prompt 是否应替换当前最优。

    Args:
        candidate_dev_score: 候选 prompt 在 dev split 上的平均得分。
        best_dev_score: 当前已接受的最优 prompt 在 dev 上的得分。
        rollback: 是否启用回滚。为 False 时无条件接受每步候选（仅用于调试）。

    Returns:
        True 表示接受候选并更新 best_prompt；False 表示回滚到原最优。
    """
    if not rollback:
        # 关闭 rollback 时始终接受，便于观察每步改动效果（不适合正式优化）
        return True
    # dev 得分持平或提升才接受；下降则拒绝，避免过拟合 train 导致 dev 变差
    return candidate_dev_score >= best_dev_score
