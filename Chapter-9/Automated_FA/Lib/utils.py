import os
import json
import random
from openai import AzureOpenAI
from tqdm import tqdm

from Lib.time_context import format_time_context_for_eval
# --- Helper Functions ---

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _make_api_call(client, model, messages, max_tokens):
    """Makes an API call to Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def _is_error_response(answer: str) -> bool:
    """判断逐步评估结果是否判定为「有错」。"""
    text = answer.strip().lower().replace(" ", "")
    return (
        text.startswith("1.yes")
        or text.startswith("1.是")
        or text.startswith("1.yes.")
        or text.startswith("1.是。")
    )


def _extract_reason(answer: str) -> str:
    """从逐步评估回复中提取原因文本。"""
    for sep in ("原因：", "原因:", "Reason:", "Reason："):
        if sep in answer:
            return answer.split(sep, 1)[-1].strip()
    return answer.strip()

# --- All-at-Once Method ---

def all_at_once(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    """
    print("\n--- Starting All-at-Once Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])

        time_context = format_time_context_for_eval()
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"{time_context}\n\n"
            f"The problem is:  {problem}\n"
            f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used in prompt
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
        )

        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        result = _make_api_call(client, model, messages, max_tokens)

        print(f"Prediction for {json_file}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")

# --- Step-by-Step Method ---

def step_by_step(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- 开始逐步归因分析 ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"跳过 {json_file}：未找到对话历史。")
            continue

        print(f"--- 正在分析：{json_file} ---")
        time_context = format_time_context_for_eval()
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"步骤 {idx} - {agent_name}: {content}\n"

            prompt = (
                "你是一名多智能体对话评估助手，负责逐步判断多智能体协作解决真实问题时，某一步是否出现会影响最终正确答案的错误。\n"
                f"{time_context}\n\n"
                f"用户问题：{problem}\n"
                f"期望答案/评判标准：{ground_truth}\n"
                f"截至当前步骤的对话历史：\n{current_conversation_history}\n"
                f"当前评估的是第 {idx} 步，执行者为「{agent_name}」。\n"
                f"请判断：第 {idx} 步是否存在明显错误，可能导致问题无法正确解决？\n"
                "注意：不要过度苛责，只关注会实质性影响结论的错误；"
                " 判断日期是否合理时必须使用上文「系统时间锚点」，不得自行假设当前年份。\n"
                "请用中文回答，且严格按以下格式输出：\n"
                "1. 是/否\n"
                "2. 原因：[你的中文解释]"
            )

            messages=[
                {"role": "system", "content": "你是严谨的多智能体对话逐步评估助手，请始终用中文回答。评估时必须遵守用户消息中的系统时间锚点，禁止臆造 2024/2025 等未在轨迹中出现的日期。"},
                {"role": "user", "content": prompt},
            ]

            print(f"正在评估第 {idx} 步（{agent_name}）...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("本步骤 API 调用失败，停止分析该文件。")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"模型判定：{answer}")

            if _is_error_response(answer):
                print(f"\nPrediction for {json_file}: 发现错误")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"错误原因: {_extract_reason(answer)}")
                error_found = True
                break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no") or answer.strip().startswith("1. 否"):
                print("本步骤未发现明显错误。")
            else:
                print("警告：模型返回格式异常，继续评估下一步。")

        if not error_found:
            print(f"\n逐步分析未在 {json_file} 中发现明确错误。")

        print("\n" + "="*50 + "\n")


# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc, time_context: str = ""):
    """Constructs the prompt for the binary search step."""
    time_block = f"{time_context}\n\n" if time_context else ""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"{time_block}"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
    )

def _report_binary_search_error(chat_history, step, json_file, is_handcrafted):
    """Reports the identified error step from binary search."""
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_recursive(client: AzureOpenAI, model: str, max_tokens: int, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool, time_context: str = ""):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error(chat_history, end if end >= 0 else 0, json_file, is_handcrafted) # Report something reasonable
         return
    if start == end:
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Cannot proceed.")
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2 

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc, time_context)

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]

    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = _make_api_call(client, model, messages, max_tokens)

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted, time_context)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, time_context)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if random.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted, time_context)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, time_context)


def binary_search(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history using a binary search approach to find the error step.
    """
    print("\n--- Starting Binary Search Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        time_context = format_time_context_for_eval()
        _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, json_file, is_handcrafted, time_context)
