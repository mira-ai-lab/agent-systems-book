"""A2A server for Hotel Recommendation Agent. Run: python -m agents.hotel_recommendation_agent.server --host 0.0.0.0 --port 9012"""

import logging
import sys

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender, InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

try:
    from .agent import HotelRecommendationAgent
    from .executor import HotelRecommendationAgentExecutor
except ImportError:  # pragma: no cover
    import os

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _AGENTS_DIR = os.path.dirname(_SCRIPT_DIR)
    _PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from agents.hotel_recommendation_agent.agent import HotelRecommendationAgent  # type: ignore
    from agents.hotel_recommendation_agent.executor import HotelRecommendationAgentExecutor  # type: ignore


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="10.112.57.99")
@click.option("--port", "port", default=9012, type=int)
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skills = [
            AgentSkill(
                id="recommend_hotel",
                name="Recommend a hotel (single)",
                description="Recommend hotels and accommodation based on a city and optional constraints. Only one recommendation per request.",
                tags=["hotel", "recommendation", "city"],
                examples=["Recommend one hotel in Shanghai under 500 CNY/night", "Recommend one hotel in Beijing for 2026-05-01 to 2026-05-03"],
            )
        ]
        agent_card = AgentCard(
            name="HotelRecommendationAgent",
            description="Recommend a single hotel based on city and preferences",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=HotelRecommendationAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=HotelRecommendationAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client=httpx_client, config_store=push_config_store)
        request_handler = DefaultRequestHandler(
            agent_executor=HotelRecommendationAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        uvicorn.run(A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler).build(), host=host, port=port)
    except Exception as e:
        logger.error("Server startup error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

