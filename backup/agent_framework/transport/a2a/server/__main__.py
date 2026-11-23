"""CLI：python -m agent_framework.transport.a2a.server --domain demo --agent EchoAgent"""

from __future__ import annotations

import click

from agent_framework.transport.a2a.server.serve import serve_sub_agent


@click.command()
@click.option("--domain", required=True, help="已注册领域名，如 demo / travel")
@click.option("--agent", "registry_agent", default=None, help="registry 工厂名，如 EchoAgent")
@click.option("--node-name", default=None, help="Supervisor 节点名，如 echo_agent")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=9012, type=int)
def main(
    domain: str,
    registry_agent: str | None,
    node_name: str | None,
    host: str,
    port: int,
) -> None:
    serve_sub_agent(
        domain,
        registry_agent=registry_agent,
        node_name=node_name,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
