"""Model-generated travel-plan drafts for the Chapter 4 case.

This module extends the capability-based task plan with a reader-facing
candidate itinerary. The model proposes activities and estimates, while code
validates the three-day structure, required goals, walking ranges, budget
ranges, and conflict-driven changes. Model estimates are never treated as
live weather, map, ticket, or price data.

:author: wulin
:date: 2026-11-21
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from llm import invoke_json
from dependency_analysis import normalize_string_list, require_text


REQUIRED_GOAL_TAGS = ["故宫", "长城", "科技类场馆"]
REQUIRED_BUDGET_CATEGORIES = ["住宿", "餐饮", "景点门票", "市内交通"]
MODEL_ESTIMATE_SCOPE = "model_generated_estimate"
PENDING_INPUT_RULES = [
    (
        ["出行日期待确认", "具体出行日期待确认"],
        {
            "field": "travel_dates",
            "reason": "需要具体日期才能查询天气、开放时间和票务",
            "affects": ["weather", "opening_hours", "tickets"],
        },
    ),
    (
        ["出发城市待确认", "出发地待确认"],
        {
            "field": "departure_city",
            "reason": "需要出发地才能估算进出北京交通",
            "affects": ["intercity_transport", "budget"],
        },
    ),
    (
        ["儿童年龄待确认"],
        {
            "field": "child_age",
            "reason": "儿童年龄会影响场馆适配、票价和活动安排",
            "affects": ["attractions", "tickets", "budget"],
        },
    ),
    (
        ["人数待确认", "入住人数待确认", "用餐人数待确认"],
        {
            "field": "party_size",
            "reason": "需要确认人数才能复核住宿、餐饮、票务和交通费用",
            "affects": ["hotel", "meals", "tickets", "budget"],
        },
    ),
]


TRAVEL_PLAN_GENERATION_SYSTEM_PROMPT = """你是一个旅行计划生成器，负责把已经完成任务分解和依赖排序的任务级计划转化为便于用户阅读的候选行程。

# 输入
1. task_plan：包含用户请求、总体目标、子任务、依赖关系和执行顺序的任务级计划。
2. trip_days：行程天数。
3. budget_limit_yuan：总预算上限。
4. walking_limit_steps：每日步行偏好上限。
5. required_goal_tags：候选行程必须覆盖的目标标签。

# 处理要求
1. 生成完整的逐日候选行程，每天给出主题、活动、市内交通建议、步数估计和雨天备选。
2. 每项活动使用 goal_tags 标注它覆盖的必去目标；故宫、长城和科技类场馆必须分别使用输入中给出的精确标签。
3. 缺少具体日期、出发城市、儿童年龄、人数或分项预算时，不得自行声称已查询真实信息，应写入 pending_inputs，并在 assumptions 中说明候选方案的估计边界。
   pending_inputs.field 优先使用 travel_dates、departure_city、child_age、party_size、accommodation_preference、cuisine_preference、sub_budget_allocation 或 mobility_needs，不要为同一含义另造字段名。
4. 步数和金额只能作为模型估计，basis 必须为 model_estimate。不得声称已调用天气、地图、票务、酒店或价格服务。
5. 预算项目只填写分项区间，不要计算总额；程序会重新计算。无法估计的费用写入 excluded_items。
6. 不得输出固定模板说明或分析过程，只输出合法 JSON。

# 输出
{
  "data_scope": "model_generated_estimate",
  "pending_inputs": [
    {
      "field": "travel_dates",
      "reason": "查询天气和开放时间需要具体日期",
      "affects": ["weather", "opening_hours"]
    }
  ],
  "assumptions": ["行程为候选方案，外部信息仍需查询核验"],
  "days": [
    {
      "day": 1,
      "theme": "当日主题",
      "activities": [
        {
          "period": "上午",
          "name": "活动名称",
          "kind": "mixed",
          "goal_tags": ["故宫"],
          "notes": ["预约或执行提示"]
        }
      ],
      "transportation": ["市内交通建议"],
      "walking_steps": {
        "min": 3000,
        "max": 6000,
        "basis": "model_estimate"
      },
      "rainy_day_backup": {
        "replacement": "雨天替代安排",
        "kind": "indoor",
        "reason": "替代原因"
      }
    }
  ],
  "budget": {
    "items": [
      {
        "category": "住宿",
        "min_yuan": 1000,
        "max_yuan": 2000,
        "basis": "model_estimate"
      },
      {
        "category": "餐饮",
        "min_yuan": 800,
        "max_yuan": 1200,
        "basis": "model_estimate"
      },
      {
        "category": "景点门票",
        "min_yuan": 300,
        "max_yuan": 600,
        "basis": "model_estimate"
      },
      {
        "category": "市内交通",
        "min_yuan": 300,
        "max_yuan": 800,
        "basis": "model_estimate"
      }
    ],
    "excluded_items": ["尚无法估计的费用"],
    "notes": ["预算说明"]
  },
  "summary": "候选行程摘要"
}

- days 必须恰好覆盖第1天到第 trip_days 天，每天只出现一次。
- kind 只能是 indoor、outdoor 或 mixed。
- required_goal_tags 中每个标签都必须出现，且只能安排在一个日期。
- transportation、activities 和 rainy_day_backup 不得为空。
- rainy_day_backup.kind 必须为 indoor。
- budget.items 必须分别包含住宿、餐饮、景点门票和市内交通四类，不得合并或遗漏；无法估算的进出北京交通写入 excluded_items。
- 金额和步数均使用非负整数区间，且下界不得大于上界。

请严格按照上述 JSON 格式返回，除此之外不要有其他输出。
"""


TRAVEL_PLAN_GENERATION_USER_PROMPT = """task_plan:
{task_plan}

trip_days: {trip_days}
budget_limit_yuan: {budget_limit_yuan}
walking_limit_steps: {walking_limit_steps}
required_goal_tags: {required_goal_tags}

请生成满足任务级计划和约束的候选旅行行程。"""


TRAVEL_PLAN_REVISION_SYSTEM_PROMPT = """你是一个旅行计划修订器，负责根据已经校验的任务级重规划结果，对候选行程进行最小必要修改。

# 输入
1. initial_itinerary：冲突发生前的完整候选行程。
2. replanning_result：已经通过程序校验的任务级重规划结果。
3. conflict_event：使原候选行程失效的运行时事件。
4. budget_limit_yuan：总预算上限。
5. walking_limit_steps：每日步行偏好上限。

# 处理要求
1. 只修改受冲突影响的日期，未受影响日期必须保持完全相同。
2. 返回修订后的完整行程，不得只返回差异。
3. 暴雨且闭园的日期必须改为纯室内安排，该日每项活动的 kind 都必须为 indoor；不得用 mixed 或 outdoor 活动替代长城。
4. 保留原候选行程中的待确认信息和估计边界；新增缺失信息可以补充，但不得声称已经查询实时外部数据。
5. 步数和金额仍是模型估计，basis 必须为 model_estimate。
6. 修订后的 days 与 budget 必须完整，具体变化日期由程序从两份行程中计算。
7. 预算项目只填写分项区间，不要计算总额；程序会重新计算。

# 输出
返回字段完整的三日候选行程，字段结构与 initial_itinerary 相同，并在顶层增加如下 revision 对象。days 和 budget.items 必须保留完整非空内容，不得使用空数组占位。
{
  "revision": {
    "event_summary": "冲突摘要",
    "change_summary": "本次行程调整摘要"
  }
}

- 只允许修改原长城日以及新长城日，其他日期必须与 initial_itinerary 完全一致。
- 长城必须离开暴雨且闭园的日期，并且修订后仍只安排在一天。
- 故宫、长城、科技类场馆三个必去标签在修订后仍必须各只出现在唯一一天；调整长城日期时，不得在另一天重复安排科技类场馆或其他必去标签。
- 暴雨且闭园日期的全部活动必须为 indoor，直接采用该日已有的雨天备选（rainy_day_backup）作为替代；不得从其他日期调入场馆，也不得改动除原长城日和新长城日以外的任何日期。

请严格输出合法 JSON，除此之外不要有其他输出。
"""


TRAVEL_PLAN_REVISION_USER_PROMPT = """initial_itinerary:
{initial_itinerary}

replanning_result:
{replanning_result}

conflict_event:
{conflict_event}

budget_limit_yuan: {budget_limit_yuan}
walking_limit_steps: {walking_limit_steps}

请根据任务级重规划结果修订候选行程。"""




def _require_non_negative_int(value: Any, field_name: str) -> int:
    """Validate one non-negative integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    """Validate one positive integer, excluding booleans."""
    number = _require_non_negative_int(value, field_name)
    if number == 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return number




def _normalize_pending_inputs(value: Any) -> List[Dict[str, Any]]:
    """Validate missing-input descriptions returned by the model."""
    if not isinstance(value, list):
        raise ValueError("pending_inputs must be a list")

    normalized: List[Dict[str, Any]] = []
    fields: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"pending_inputs item {index} must be an object")
        field = require_text(item.get("field"), f"pending_inputs[{index}].field")
        if field in fields:
            raise ValueError(f"pending_inputs contains duplicate field: {field}")
        fields.add(field)
        normalized.append(
            {
                "field": field,
                "reason": require_text(
                    item.get("reason"),
                    f"pending_inputs[{index}].reason",
                ),
                "affects": normalize_string_list(
                    item.get("affects", []),
                    f"pending_inputs[{index}].affects",
                ),
            }
        )
    return normalized


def _normalize_activity(value: Any, day: int, index: int) -> Dict[str, Any]:
    """Validate one itinerary activity."""
    field_name = f"days[{day}].activities[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    kind = require_text(value.get("kind"), f"{field_name}.kind")
    if kind not in {"indoor", "outdoor", "mixed"}:
        raise ValueError(f"{field_name}.kind must be indoor, outdoor, or mixed")
    return {
        "period": require_text(value.get("period"), f"{field_name}.period"),
        "name": require_text(value.get("name"), f"{field_name}.name"),
        "kind": kind,
        "goal_tags": normalize_string_list(
            value.get("goal_tags", []),
            f"{field_name}.goal_tags",
        ),
        "notes": normalize_string_list(
            value.get("notes", []),
            f"{field_name}.notes",
        ),
    }


def _normalize_day(value: Any, index: int) -> Dict[str, Any]:
    """Validate one complete itinerary day."""
    if not isinstance(value, dict):
        raise ValueError(f"days item {index} must be an object")
    day = _require_positive_int(value.get("day"), f"days[{index}].day")
    activities_value = value.get("activities")
    if not isinstance(activities_value, list) or not activities_value:
        raise ValueError(f"days[{day}].activities must be a non-empty list")
    activities = [
        _normalize_activity(activity, day, activity_index)
        for activity_index, activity in enumerate(activities_value, start=1)
    ]

    walking = value.get("walking_steps")
    if not isinstance(walking, dict):
        raise ValueError(f"days[{day}].walking_steps must be an object")
    minimum = _require_non_negative_int(
        walking.get("min"),
        f"days[{day}].walking_steps.min",
    )
    maximum = _require_non_negative_int(
        walking.get("max"),
        f"days[{day}].walking_steps.max",
    )
    if minimum > maximum:
        raise ValueError(f"days[{day}].walking_steps min cannot exceed max")
    basis = require_text(
        walking.get("basis"),
        f"days[{day}].walking_steps.basis",
    )
    if basis != "model_estimate":
        raise ValueError(f"days[{day}].walking_steps.basis must be model_estimate")

    backup = value.get("rainy_day_backup")
    if not isinstance(backup, dict):
        raise ValueError(f"days[{day}].rainy_day_backup must be an object")
    backup_kind = require_text(
        backup.get("kind"),
        f"days[{day}].rainy_day_backup.kind",
    )
    if backup_kind != "indoor":
        raise ValueError(f"days[{day}].rainy_day_backup.kind must be indoor")
    transportation = normalize_string_list(
        value.get("transportation"),
        f"days[{day}].transportation",
    )
    if not transportation:
        raise ValueError(f"days[{day}].transportation must not be empty")
    return {
        "day": day,
        "theme": require_text(value.get("theme"), f"days[{day}].theme"),
        "activities": activities,
        "transportation": transportation,
        "walking_steps": {
            "min": minimum,
            "max": maximum,
            "basis": basis,
        },
        "rainy_day_backup": {
            "replacement": require_text(
                backup.get("replacement"),
                f"days[{day}].rainy_day_backup.replacement",
            ),
            "kind": backup_kind,
            "reason": require_text(
                backup.get("reason"),
                f"days[{day}].rainy_day_backup.reason",
            ),
        },
    }


def _normalize_budget(
    value: Any,
    pending_inputs: List[Dict[str, Any]],
    budget_limit_yuan: int,
) -> Dict[str, Any]:
    """Validate budget ranges and calculate the aggregate range in code."""
    if not isinstance(value, dict):
        raise ValueError("budget must be an object")
    raw_items = value.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("budget.items must be a non-empty list")

    items: List[Dict[str, Any]] = []
    categories: set[str] = set()
    total_min = 0
    total_max = 0
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"budget.items item {index} must be an object")
        category = require_text(
            item.get("category"),
            f"budget.items[{index}].category",
        )
        if category in categories:
            raise ValueError(f"budget.items contains duplicate category: {category}")
        categories.add(category)
        minimum = _require_non_negative_int(
            item.get("min_yuan"),
            f"budget.items[{index}].min_yuan",
        )
        maximum = _require_non_negative_int(
            item.get("max_yuan"),
            f"budget.items[{index}].max_yuan",
        )
        if minimum > maximum:
            raise ValueError(
                f"budget.items[{index}] min_yuan cannot exceed max_yuan"
            )
        basis = require_text(
            item.get("basis"),
            f"budget.items[{index}].basis",
        )
        if basis != "model_estimate":
            raise ValueError(
                f"budget.items[{index}].basis must be model_estimate"
            )
        items.append(
            {
                "category": category,
                "min_yuan": minimum,
                "max_yuan": maximum,
                "basis": basis,
            }
        )
        total_min += minimum
        total_max += maximum

    excluded_items = normalize_string_list(
        value.get("excluded_items", []),
        "budget.excluded_items",
    )
    notes = normalize_string_list(value.get("notes", []), "budget.notes")
    missing_categories = [
        category
        for category in REQUIRED_BUDGET_CATEGORIES
        if category not in categories
    ]
    if missing_categories:
        raise ValueError(
            "budget.items is missing required categories: "
            f"{missing_categories}"
        )
    if pending_inputs or excluded_items:
        status = "pending_external_verification"
    elif total_max <= budget_limit_yuan:
        status = "estimated_within_limit"
    elif total_min > budget_limit_yuan:
        status = "estimated_over_limit"
    else:
        status = "may_exceed_limit"
    return {
        "limit_yuan": budget_limit_yuan,
        "items": items,
        "estimated_total_yuan": {
            "min": total_min,
            "max": total_max,
        },
        "excluded_items": excluded_items,
        "notes": notes,
        "status": status,
        "basis": "model_estimate",
    }


def _normalize_travel_plan_payload(
    payload: Dict[str, Any],
    *,
    trip_days: int,
    budget_limit_yuan: int,
    walking_limit_steps: int,
    version: int,
    status: str,
) -> Dict[str, Any]:
    """Normalize common candidate-itinerary fields."""
    if not isinstance(payload, dict):
        raise ValueError("Travel plan response must be a JSON object")
    scope = require_text(payload.get("data_scope"), "data_scope")
    if scope != MODEL_ESTIMATE_SCOPE:
        raise ValueError(f"data_scope must be {MODEL_ESTIMATE_SCOPE}")

    pending_inputs = _normalize_pending_inputs(payload.get("pending_inputs", []))
    assumptions = normalize_string_list(
        payload.get("assumptions", []),
        "assumptions",
    )
    raw_days = payload.get("days")
    if not isinstance(raw_days, list):
        raise ValueError("days must be a list")
    days = [
        _normalize_day(day, index)
        for index, day in enumerate(raw_days, start=1)
    ]
    actual_day_numbers = [day["day"] for day in days]
    expected_day_numbers = list(range(1, trip_days + 1))
    if sorted(actual_day_numbers) != expected_day_numbers:
        raise ValueError(
            "days must exactly cover the requested trip; "
            f"expected={expected_day_numbers}, actual={actual_day_numbers}"
        )
    days.sort(key=lambda item: item["day"])

    goal_tag_days = {
        tag: sorted(
            {
                day["day"]
                for day in days
                for activity in day["activities"]
                if tag in activity["goal_tags"]
            }
        )
        for tag in REQUIRED_GOAL_TAGS
    }
    missing_tags = [tag for tag, tag_days in goal_tag_days.items() if not tag_days]
    if missing_tags:
        raise ValueError(f"Travel plan is missing required goal tags: {missing_tags}")
    repeated_tags = {
        tag: tag_days
        for tag, tag_days in goal_tag_days.items()
        if len(tag_days) > 1
    }
    if repeated_tags:
        raise ValueError(
            "Each required goal tag must appear on exactly one day: "
            f"{repeated_tags}"
        )

    days_exceeding_limit = [
        day["day"]
        for day in days
        if day["walking_steps"]["max"] > walking_limit_steps
    ]
    walking_status = (
        "within_preference" if not days_exceeding_limit else "needs_adjustment"
    )
    budget = _normalize_budget(
        payload.get("budget"),
        pending_inputs,
        budget_limit_yuan,
    )
    return {
        "version": version,
        "status": status,
        "data_scope": scope,
        "trip_days": trip_days,
        "pending_inputs": pending_inputs,
        "assumptions": assumptions,
        "days": days,
        "budget": budget,
        "constraint_check": {
            "required_goal_tags": {
                "status": "covered",
                "required": list(REQUIRED_GOAL_TAGS),
                "scheduled_days": {
                    tag: tag_days[0]
                    for tag, tag_days in goal_tag_days.items()
                },
            },
            "walking_preference": {
                "status": walking_status,
                "limit_steps": walking_limit_steps,
                "days_exceeding_limit": days_exceeding_limit,
                "basis": "model_estimate",
            },
            "budget": {
                "status": budget["status"],
                "limit_yuan": budget_limit_yuan,
                "basis": "model_estimate",
            },
        },
        "summary": require_text(payload.get("summary"), "summary"),
    }


def _pending_inputs_from_task_plan(
    task_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Recover explicitly unresolved inputs from the validated task plan."""
    text = json.dumps(task_plan, ensure_ascii=False)
    return [
        copy.deepcopy(item)
        for markers, item in PENDING_INPUT_RULES
        if any(marker in text for marker in markers)
    ]


def _merge_required_pending_inputs(
    payload: Dict[str, Any],
    required_pending_inputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep task-plan pending inputs even if the model omits them."""
    if not isinstance(payload, dict):
        raise ValueError("Travel plan response must be a JSON object")
    merged = copy.deepcopy(payload)
    proposed = merged.get("pending_inputs", [])
    if not isinstance(proposed, list):
        raise ValueError("pending_inputs must be a list")
    pending = copy.deepcopy(required_pending_inputs)
    known_fields = {
        item.get("field")
        for item in pending
        if isinstance(item, dict)
    }
    for item in proposed:
        if isinstance(item, dict) and item.get("field") in known_fields:
            continue
        pending.append(copy.deepcopy(item))
        if isinstance(item, dict):
            known_fields.add(item.get("field"))
    merged["pending_inputs"] = pending
    return merged


def build_travel_plan_prompt(
    task_plan: Dict[str, Any],
    *,
    trip_days: int,
    budget_limit_yuan: int,
    walking_limit_steps: int,
) -> str:
    """Build the user prompt for one model-generated candidate itinerary."""
    if not isinstance(task_plan, dict) or not task_plan:
        raise ValueError("task_plan must be a non-empty object")
    days = _require_positive_int(trip_days, "trip_days")
    budget_limit = _require_positive_int(
        budget_limit_yuan,
        "budget_limit_yuan",
    )
    walking_limit = _require_positive_int(
        walking_limit_steps,
        "walking_limit_steps",
    )
    return TRAVEL_PLAN_GENERATION_USER_PROMPT.format(
        task_plan=json.dumps(task_plan, ensure_ascii=False, indent=2),
        trip_days=days,
        budget_limit_yuan=budget_limit,
        walking_limit_steps=walking_limit,
        required_goal_tags=json.dumps(REQUIRED_GOAL_TAGS, ensure_ascii=False),
    )


def normalize_travel_plan_response(
    payload: Dict[str, Any],
    *,
    trip_days: int,
    budget_limit_yuan: int,
    walking_limit_steps: int,
    required_pending_inputs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Validate one model-generated candidate itinerary."""
    normalized_payload = _merge_required_pending_inputs(
        payload,
        required_pending_inputs or [],
    )
    return _normalize_travel_plan_payload(
        normalized_payload,
        trip_days=_require_positive_int(trip_days, "trip_days"),
        budget_limit_yuan=_require_positive_int(
            budget_limit_yuan,
            "budget_limit_yuan",
        ),
        walking_limit_steps=_require_positive_int(
            walking_limit_steps,
            "walking_limit_steps",
        ),
        version=1,
        status="draft",
    )


async def generate_travel_plan(
    llm: Any,
    task_plan: Dict[str, Any],
    *,
    trip_days: int,
    budget_limit_yuan: int,
    walking_limit_steps: int,
) -> Dict[str, Any]:
    """Generate and validate one reader-facing candidate itinerary.

    Args:
        llm: LangChain-compatible chat model.
        task_plan: Validated task-level plan from the earlier planning stages.
        trip_days: Number of itinerary days.
        budget_limit_yuan: User-provided total budget limit.
        walking_limit_steps: Preferred maximum walking steps per day.

    Returns:
        A normalized candidate itinerary containing model-estimate markers.
    """
    prompt = build_travel_plan_prompt(
        task_plan,
        trip_days=trip_days,
        budget_limit_yuan=budget_limit_yuan,
        walking_limit_steps=walking_limit_steps,
    )
    payload = await invoke_json(
        llm,
        prompt,
        system_prompt=TRAVEL_PLAN_GENERATION_SYSTEM_PROMPT,
        require_object=True,
    )
    return normalize_travel_plan_response(
        payload,
        trip_days=trip_days,
        budget_limit_yuan=budget_limit_yuan,
        walking_limit_steps=walking_limit_steps,
        required_pending_inputs=_pending_inputs_from_task_plan(task_plan),
    )


def find_day_by_goal_tag(itinerary: Dict[str, Any], goal_tag: str) -> int:
    """Return the unique itinerary day containing one exact goal tag."""
    if not isinstance(itinerary, dict):
        raise ValueError("itinerary must be an object")
    tag = require_text(goal_tag, "goal_tag")
    days = itinerary.get("days")
    if not isinstance(days, list):
        raise ValueError("itinerary.days must be a list")
    matching_days: List[int] = []
    for day in days:
        if not isinstance(day, dict):
            continue
        activities = day.get("activities", [])
        if not isinstance(activities, list):
            continue
        if any(
            isinstance(activity, dict)
            and tag in activity.get("goal_tags", [])
            for activity in activities
        ):
            matching_days.append(
                _require_positive_int(day.get("day"), "itinerary day")
            )
    unique_days = sorted(set(matching_days))
    if not unique_days:
        raise ValueError(f"No itinerary day covers goal tag: {tag}")
    if len(unique_days) != 1:
        raise ValueError(
            f"Goal tag {tag} must appear on exactly one day: {unique_days}"
        )
    return unique_days[0]


def build_rain_conflict(wall_day: int, *, trip_days: int) -> str:
    """Build a runtime rain-and-closure event from the generated itinerary."""
    day = _require_positive_int(wall_day, "wall_day")
    total_days = _require_positive_int(trip_days, "trip_days")
    if day > total_days:
        raise ValueError("wall_day must be within the trip")
    other_days = [value for value in range(1, total_days + 1) if value != day]
    other_day_text = "、".join(f"第{value}天" for value in other_days)
    return (
        f"运行时收到新信息：候选行程将长城安排在第{day}天，但该日北京山区"
        "出现暴雨，长城景区当天临时关闭；"
        f"{other_day_text}适合户外活动，故宫和科技类场馆正常开放，"
        "该日应调整为纯室内项目；市内交通、住宿和餐饮服务正常。"
        "用户仍要求保留原旅行天数、"
        "预算上限以及故宫、长城和科技类场馆三个必去项目，并希望优先"
        "通过调整日期顺序解决。"
    )


def build_travel_revision_prompt(
    initial_itinerary: Dict[str, Any],
    replanning_result: Dict[str, Any],
    conflict_event: str,
    *,
    budget_limit_yuan: int,
    walking_limit_steps: int,
) -> str:
    """Build the user prompt for conflict-driven itinerary revision."""
    if not isinstance(initial_itinerary, dict) or not initial_itinerary:
        raise ValueError("initial_itinerary must be a non-empty object")
    if not isinstance(replanning_result, dict):
        raise ValueError("replanning_result must be an object")
    if replanning_result.get("status") != "replanned":
        raise ValueError("replanning_result.status must be replanned")
    event = require_text(conflict_event, "conflict_event")
    budget_limit = _require_positive_int(
        budget_limit_yuan,
        "budget_limit_yuan",
    )
    walking_limit = _require_positive_int(
        walking_limit_steps,
        "walking_limit_steps",
    )
    return TRAVEL_PLAN_REVISION_USER_PROMPT.format(
        initial_itinerary=json.dumps(
            initial_itinerary,
            ensure_ascii=False,
            indent=2,
        ),
        replanning_result=json.dumps(
            replanning_result,
            ensure_ascii=False,
            indent=2,
        ),
        conflict_event=event,
        budget_limit_yuan=budget_limit,
        walking_limit_steps=walking_limit,
    )


def _day_overview(day: Dict[str, Any]) -> str:
    """Build a compact, deterministic description of one itinerary day."""
    activity_names = [activity["name"] for activity in day["activities"]]
    return "、".join(activity_names)


def _merge_revision_context(
    payload: Dict[str, Any],
    initial_itinerary: Dict[str, Any],
) -> Dict[str, Any]:
    """Preserve unresolved context when normalizing a model revision."""
    if not isinstance(payload, dict):
        raise ValueError("Travel revision response must be a JSON object")
    merged = copy.deepcopy(payload)
    merged.setdefault("data_scope", initial_itinerary.get("data_scope"))

    proposed_pending = merged.get("pending_inputs", [])
    if not isinstance(proposed_pending, list):
        raise ValueError("pending_inputs must be a list")
    pending = copy.deepcopy(initial_itinerary.get("pending_inputs", []))
    known_fields = {
        item.get("field")
        for item in pending
        if isinstance(item, dict)
    }
    for item in proposed_pending:
        if isinstance(item, dict) and item.get("field") in known_fields:
            continue
        pending.append(copy.deepcopy(item))
        if isinstance(item, dict):
            known_fields.add(item.get("field"))
    merged["pending_inputs"] = pending

    proposed_assumptions = merged.get("assumptions", [])
    if not isinstance(proposed_assumptions, list):
        raise ValueError("assumptions must be a list")
    assumptions = copy.deepcopy(initial_itinerary.get("assumptions", []))
    for item in proposed_assumptions:
        if item not in assumptions:
            assumptions.append(copy.deepcopy(item))
    merged["assumptions"] = assumptions

    budget = merged.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("budget must be an object")
    proposed_excluded = budget.get("excluded_items", [])
    if not isinstance(proposed_excluded, list):
        raise ValueError("budget.excluded_items must be a list")
    initial_budget = initial_itinerary.get("budget", {})
    excluded = copy.deepcopy(initial_budget.get("excluded_items", []))
    for item in proposed_excluded:
        if item not in excluded:
            excluded.append(copy.deepcopy(item))
    budget["excluded_items"] = excluded

    proposed_notes = budget.get("notes", [])
    if not isinstance(proposed_notes, list):
        raise ValueError("budget.notes must be a list")
    notes = copy.deepcopy(initial_budget.get("notes", []))
    for item in proposed_notes:
        if item not in notes:
            notes.append(copy.deepcopy(item))
    budget["notes"] = notes
    merged["budget"] = budget
    return merged


def normalize_travel_revision_response(
    payload: Dict[str, Any],
    initial_itinerary: Dict[str, Any],
    conflict_event: str,
    *,
    budget_limit_yuan: int,
    walking_limit_steps: int,
    blocked_day: int | None = None,
    blocked_goal_tag: str = "长城",
) -> Dict[str, Any]:
    """Validate a model revision and prove that the live conflict is resolved."""
    if not isinstance(initial_itinerary, dict):
        raise ValueError("initial_itinerary must be an object")
    initial_days = initial_itinerary.get("days")
    if not isinstance(initial_days, list) or not initial_days:
        raise ValueError("initial_itinerary.days must be a non-empty list")
    trip_days = _require_positive_int(
        initial_itinerary.get("trip_days"),
        "initial_itinerary.trip_days",
    )
    try:
        current_version = int(initial_itinerary.get("version", 1))
    except (TypeError, ValueError) as error:
        raise ValueError("initial_itinerary.version must be an integer") from error
    event = require_text(conflict_event, "conflict_event")
    goal_tag = require_text(blocked_goal_tag, "blocked_goal_tag")
    initial_goal_day = find_day_by_goal_tag(initial_itinerary, goal_tag)
    conflict_day = (
        initial_goal_day
        if blocked_day is None
        else _require_positive_int(blocked_day, "blocked_day")
    )
    if conflict_day != initial_goal_day:
        raise ValueError(
            "blocked_day must be the original day containing the blocked goal; "
            f"expected={initial_goal_day}, actual={conflict_day}"
        )

    merged_payload = _merge_revision_context(payload, initial_itinerary)
    revised = _normalize_travel_plan_payload(
        merged_payload,
        trip_days=trip_days,
        budget_limit_yuan=_require_positive_int(
            budget_limit_yuan,
            "budget_limit_yuan",
        ),
        walking_limit_steps=_require_positive_int(
            walking_limit_steps,
            "walking_limit_steps",
        ),
        version=current_version + 1,
        status="revised",
    )
    revision = merged_payload.get("revision")
    if not isinstance(revision, dict):
        raise ValueError("revision must be an object")
    initial_by_day = {day["day"]: day for day in initial_days}
    revised_by_day = {day["day"]: day for day in revised["days"]}
    actual_changed_days = [
        day
        for day in range(1, trip_days + 1)
        if initial_by_day.get(day) != revised_by_day.get(day)
    ]
    if not actual_changed_days:
        raise ValueError("Revised itinerary must change at least one day")
    revised_goal_day = find_day_by_goal_tag(revised, goal_tag)
    if revised_goal_day == conflict_day:
        raise ValueError(
            f"Revised itinerary still places {goal_tag} on blocked day "
            f"{conflict_day}"
        )
    expected_changed_days = sorted({conflict_day, revised_goal_day})
    if actual_changed_days != expected_changed_days:
        raise ValueError(
            "Revision must make only the minimum day changes needed to move "
            f"{goal_tag}; expected={expected_changed_days}, "
            f"actual={actual_changed_days}"
        )

    non_indoor_activities = [
        activity["name"]
        for activity in revised_by_day[conflict_day]["activities"]
        if activity["kind"] != "indoor"
    ]
    if non_indoor_activities:
        raise ValueError(
            "Revised conflict day must contain indoor activities only: "
            f"{non_indoor_activities}"
        )

    unchanged_days = [
        day for day in range(1, trip_days + 1) if day not in actual_changed_days
    ]
    derived_changes = [
        {
            "day": day,
            "before": _day_overview(initial_by_day[day]),
            "after": _day_overview(revised_by_day[day]),
            "reason": (
                f"第{conflict_day}天原{goal_tag}安排遭遇暴雨且景区关闭，"
                "改为纯室内项目。"
                if day == conflict_day
                else f"将{goal_tag}调整到第{day}天，避开第{conflict_day}天"
                "的暴雨和闭园。"
            ),
        }
        for day in actual_changed_days
    ]
    revised["revision"] = {
        "conflict_event": event,
        "event_summary": require_text(
            revision.get("event_summary"),
            "revision.event_summary",
        ),
        "blocked_goal_tag": goal_tag,
        "blocked_day": conflict_day,
        "new_goal_day": revised_goal_day,
        "changed_days": derived_changes,
        "unchanged_days": unchanged_days,
        "change_summary": require_text(
            revision.get("change_summary"),
            "revision.change_summary",
        ),
    }
    return revised


async def revise_travel_plan(
    llm: Any,
    initial_itinerary: Dict[str, Any],
    replanning_result: Dict[str, Any],
    conflict_event: str,
    *,
    budget_limit_yuan: int,
    walking_limit_steps: int,
) -> Dict[str, Any]:
    """Generate and validate a concrete itinerary revision.

    Args:
        llm: LangChain-compatible chat model.
        initial_itinerary: Normalized candidate itinerary before the conflict.
        replanning_result: Validated task-level replanning result.
        conflict_event: Runtime event that invalidated the candidate itinerary.
        budget_limit_yuan: User-provided total budget limit.
        walking_limit_steps: Preferred maximum walking steps per day.

    Returns:
        A normalized revised itinerary with a deterministic change summary.

    Raises:
        ValueError: If task-level replanning did not return ``replanned``.
    """
    prompt = build_travel_revision_prompt(
        initial_itinerary,
        replanning_result,
        conflict_event,
        budget_limit_yuan=budget_limit_yuan,
        walking_limit_steps=walking_limit_steps,
    )
    payload = await invoke_json(
        llm,
        prompt,
        system_prompt=TRAVEL_PLAN_REVISION_SYSTEM_PROMPT,
        require_object=True,
    )
    return normalize_travel_revision_response(
        payload,
        initial_itinerary,
        conflict_event,
        budget_limit_yuan=budget_limit_yuan,
        walking_limit_steps=walking_limit_steps,
    )


def summarize_travel_plan(itinerary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact itinerary summary suitable for teaching output."""
    if not isinstance(itinerary, dict):
        raise ValueError("itinerary must be an object")
    days = itinerary.get("days")
    if not isinstance(days, list):
        raise ValueError("itinerary.days must be a list")
    return {
        "version": itinerary.get("version"),
        "status": itinerary.get("status"),
        "data_scope": itinerary.get("data_scope"),
        "pending_inputs": [
            item.get("field")
            for item in itinerary.get("pending_inputs", [])
            if isinstance(item, dict)
        ],
        "days": [
            {
                "day": day["day"],
                "theme": day["theme"],
                "activities": _day_overview(day),
                "transportation": "；".join(
                    item.rstrip("。；;")
                    for item in day["transportation"]
                ) + "。",
                "walking_steps": (
                    f"{day['walking_steps']['min']}-"
                    f"{day['walking_steps']['max']}步（模型估计）"
                ),
                "rainy_day_backup": day["rainy_day_backup"]["replacement"],
            }
            for day in days
        ],
        "budget": {
            "estimated_total_yuan": itinerary.get("budget", {}).get(
                "estimated_total_yuan"
            ),
            "status": itinerary.get("budget", {}).get("status"),
            "excluded_items": itinerary.get("budget", {}).get(
                "excluded_items",
                [],
            ),
        },
        "constraint_check": itinerary.get("constraint_check"),
        "summary": itinerary.get("summary"),
    }


def summarize_travel_changes(
    initial_itinerary: Dict[str, Any],
    revised_itinerary: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a compact before/after summary for a revised itinerary."""
    if not isinstance(initial_itinerary, dict):
        raise ValueError("initial_itinerary must be an object")
    if not isinstance(revised_itinerary, dict):
        raise ValueError("revised_itinerary must be an object")
    revision = revised_itinerary.get("revision")
    if not isinstance(revision, dict):
        raise ValueError("revised_itinerary.revision must be an object")
    return {
        "status": "replanned",
        "version_change": (
            f"{initial_itinerary.get('version')} -> "
            f"{revised_itinerary.get('version')}"
        ),
        "changed_days": [
            dict(change)
            for change in revision.get("changed_days", [])
        ],
        "unchanged_days": revision.get("unchanged_days", []),
        "change_summary": revision.get("change_summary"),
        "budget": revised_itinerary.get("budget"),
        "constraint_check": revised_itinerary.get("constraint_check"),
    }
