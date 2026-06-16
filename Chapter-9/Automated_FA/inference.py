import os
import argparse
import contextlib
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from Lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search
)


KNOWN_GPT_MODELS = {"gpt-4o", "gpt4", "gpt4o-mini"}
DASHSCOPE_MODELS = {"qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-72b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | DASHSCOPE_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}

def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

def _default_device() -> str:
    env_device = os.getenv("INFERENCE_DEVICE")
    if env_device:
        return env_device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _init_local_model(model_alias: str, device: str):
    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

    model_id = LOCAL_MODEL_MAP[model_alias]

    if model_alias in LOCAL_LLAMA_ALIASES:
        print(f"Selected local Llama model: {model_alias} ({model_id}) on device {device}")
        client_or_model_obj = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        print(f"Successfully initialized Llama pipeline on {device}.")
        return client_or_model_obj, "llama", model_id

    print(f"Selected local Qwen model: {model_alias} ({model_id}) on device {device}")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Successfully initialized Qwen model and tokenizer on {device}.")
    return (qwen_model, qwen_tokenizer), "qwen", model_id


def main():
    _load_env()
    default_device = _default_device()

    parser = argparse.ArgumentParser(description="Analyze multi-agent chat history using specific models.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default = "../Who&When/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files. Default: '../Who&When/Algorithm-Generated'."
    )

    parser.add_argument(
        "--is_handcrafted",
        type=str,
        default="False",
        choices=['True', 'False'], # If you want to test Hand-Crafted, set is_handcrafted to be True.
        help="Specify 'True' or 'False'. Default: 'False'."
    )


    parser.add_argument(
        "--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "",
        help="API Key. Azure OpenAI or DashScope (通义千问). Reads AZURE_OPENAI_API_KEY / DASHSCOPE_API_KEY from .env."
    )
    parser.add_argument(
        "--azure_endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
        help="Azure OpenAI Endpoint URL. Required only for GPT models."
    )
    parser.add_argument(
        "--base_url", type=str, default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        help="DashScope OpenAI-compatible base URL. Used for qwen-plus / qwen-max / qwen-turbo."
    )
    parser.add_argument(
        "--api_version", type=str, default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        help="Azure OpenAI API Version. Used only for GPT models."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for GPT API response. Used only for GPT models."
    )

    parser.add_argument(
        "--device", type=str, default=default_device,
        help="Device for local model inference (e.g., 'cuda', 'cuda:0', 'cpu')."
    )

    args = parser.parse_args()

    client_or_model_obj = None
    model_type = None # gpt, llama, qwen
    model_family = None 
    model_id_or_deployment = args.model

    if args.model in KNOWN_GPT_MODELS:
        model_type = 'gpt'
        model_family = 'gpt'
        print(f"Selected GPT model: {args.model}")
       
        if not args.api_key:
            print("Error: --api_key or AZURE_OPENAI_API_KEY in .env is required for GPT models")
            sys.exit(1)
        if not args.azure_endpoint:
            print("Error: --azure_endpoint or AZURE_OPENAI_ENDPOINT in .env is required for GPT models")
            sys.exit(1)
        try:
            client_or_model_obj = AzureOpenAI(
                api_key=args.api_key,
                api_version=args.api_version,
                azure_endpoint=args.azure_endpoint,
            )
            print(f"Successfully initialized AzureOpenAI client for endpoint: {args.azure_endpoint}")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
            sys.exit(1)

    elif args.model in DASHSCOPE_MODELS:
        model_type = 'gpt'
        model_family = 'dashscope'
        model_id_or_deployment = args.model
        if not args.api_key:
            print("Error: --api_key or DASHSCOPE_API_KEY in Automated_FA/.env is required for 通义千问 API models")
            sys.exit(1)
        try:
            client_or_model_obj = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url,
            )
            print(f"Successfully initialized DashScope client: {args.base_url}")
            print(f"Model: {args.model}")
        except Exception as e:
            print(f"Error initializing DashScope client: {e}")
            sys.exit(1)

    elif args.model in LOCAL_MODEL_ALIASES:
        model_type = 'local'
        try:
            client_or_model_obj, model_family, model_id_or_deployment = _init_local_model(
                args.model, args.device
            )
        except Exception as e:
            print(f"Error initializing local model {args.model}: {e}")
            print("Make sure you have sufficient VRAM/RAM and necessary libraries (transformers, torch, accelerate).")
            sys.exit(1)
    else:
        print(f"Error: Invalid model '{args.model}' specified.")
        sys.exit(1)


    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if args.is_handcrafted == "True" else "_alg_generated"
    output_filename = f"{args.method}_{args.model.replace('/','_')}{handcrafted_suffix}.txt"
    output_filepath = os.path.join(output_dir, output_filename)
    
    args.is_handcrafted = True if args.is_handcrafted == "True" else False # Update: Convert string to boolean

    print(f"Analysis method: {args.method}")
    print(f"Model Alias: {args.model} (Family: {model_family})")
    print(f"Output will be saved to: {output_filepath}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Family: {model_family}")
            print(f"Model Used: {model_id_or_deployment}")
            print(f"Input Directory: {args.directory_path}")
            print(f"Is Handcrafted: {args.is_handcrafted}")
            print("-" * 20)

            if model_type == 'gpt':
                if args.method == "all_at_once":
                    gpt_all_at_once(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "step_by_step":
                    gpt_step_by_step(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "binary_search":
                    gpt_binary_search(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
            elif model_type == 'local':
                from Lib.local_model import (
                    analyze_all_at_once_local,
                    analyze_step_by_step_local,
                    analyze_binary_search_local,
                )
                if args.method == "all_at_once":
                    analyze_all_at_once_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family
                    )
                elif args.method == "step_by_step":
                    analyze_step_by_step_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family
                    )
                elif args.method == "binary_search":
                    analyze_binary_search_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family
                    )

            else:
                 print(f"Internal Error: Unknown model_type '{model_type}' during function call.")


            print("-" * 20)
            print(f"--- Analysis Complete ---")

        print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
  
if __name__ == "__main__":
    main()