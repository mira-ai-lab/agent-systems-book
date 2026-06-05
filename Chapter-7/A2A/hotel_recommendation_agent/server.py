"""A2A server for Hotel Recommendation Agent.

Run from this directory:
  python server.py --host 0.0.0.0 --port 9012
"""

from __future__ import annotations

import logging
import sys

import click
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from dotenv import load_dotenv
from starlette.applications import Starlette

try:
    from .agent import HotelRecommendationAgent
    from .executor import HotelRecommendationAgentExecutor
except ImportError:  # pragma: no cover
    import os

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from agent import HotelRecommendationAgent
    from executor import HotelRecommendationAgentExecutor


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_agent_card(host: str, port: int) -> AgentCard:
    rpc_url = f"http://{host}:{port}/"
    return AgentCard(
        name="HotelRecommendationAgent",
        description="Recommend hotels based on city and preferences",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=rpc_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=HotelRecommendationAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=HotelRecommendationAgent.SUPPORTED_CONTENT_TYPES,
        skills=[
            AgentSkill(
                id="recommend_hotel",
                name="Recommend hotels",
                description="Recommend hotels and accommodation based on a city and optional constraints.",
                tags=["hotel", "recommendation", "city"],
                examples=[
                    "Recommend one hotel in Shanghai under 500 CNY/night",
                    "Recommend hotels near Datong ancient city, budget 500/night",
                ],
            )
        ],
    )


def _build_app(host: str, port: int) -> Starlette:
    agent_card = _build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=HotelRecommendationAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(
        create_jsonrpc_routes(
            request_handler,
            rpc_url="/",
            enable_v0_3_compat=True,  # 兼容 supervisor_local 等 v0.3 message/send 客户端
        )
    )
    return Starlette(routes=routes)


@click.command()
@click.option("--host", "host", default="127.0.0.1")
@click.option("--port", "port", default=9012, type=int)
def main(host: str, port: int) -> None:
    try:
        app = _build_app(host, port)
        logger.info("HotelRecommendationAgent listening on http://%s:%s/", host, port)
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error("Server startup error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
