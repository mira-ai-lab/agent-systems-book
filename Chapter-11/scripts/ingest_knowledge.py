"""将 domains/*/knowledge/documents.json ingest 到 data/knowledge/{domain}/ + Chroma。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_framework.config import load_project_dotenv
from agent_framework.domain.plugin_registry import list_domains
from agent_framework.router.kb.repository import ingest_domain_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest 领域知识库到 Chroma（data/knowledge/{domain}/）")
    parser.add_argument("--domain", help="领域名，如 customer_service")
    parser.add_argument(
        "--all",
        action="store_true",
        help="ingest 所有含 bundle documents.json 的已注册领域",
    )
    parser.add_argument(
        "--embedding-backend",
        default="hashing",
        help="hashing | embedding（默认 hashing）",
    )
    args = parser.parse_args()

    load_project_dotenv()

    if args.all:
        domains = [item["name"] for item in list_domains()]
    elif args.domain:
        domains = [args.domain.strip()]
    else:
        parser.error("请指定 --domain 或 --all")

    for domain in domains:
        try:
            count = ingest_domain_knowledge(domain, embedding_backend=args.embedding_backend)
        except ValueError as exc:
            print(f"[skip] {domain}: {exc}")
            continue
        print(f"[ok] {domain}: ingested {count} documents → data/knowledge/{domain}/")


if __name__ == "__main__":
    main()
