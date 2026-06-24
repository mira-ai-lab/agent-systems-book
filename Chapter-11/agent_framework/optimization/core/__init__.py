"""优化子系统公共基础模块（回滚、结果类型、持久化）。

本包为 local / textgrad_lib 等并列 Optimizer 提供统一的数据结构与工具函数，
避免各 Optimizer 重复定义结果格式与保存逻辑。
"""

from .result import OptimizationResult, OptimizationStepRecord
from .rollback import should_accept_candidate
from .save import (
    save_agent_optimization_artifacts,
    save_decomposition_optimization_artifacts,
    save_multi_agent_optimization_artifacts,
    save_planner_optimization_artifacts,
    save_routing_optimization_artifacts,
)

__all__ = [
    "OptimizationResult",
    "OptimizationStepRecord",
    "should_accept_candidate",
    "save_agent_optimization_artifacts",
    "save_multi_agent_optimization_artifacts",
    "save_decomposition_optimization_artifacts",
    "save_routing_optimization_artifacts",
    "save_planner_optimization_artifacts",
]
