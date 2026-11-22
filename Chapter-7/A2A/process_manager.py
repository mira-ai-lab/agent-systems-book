"""Manage local A2A server processes used by the teaching Notebook.

Main capabilities:

- Start Planner and Hotel A2A services on available ports.
- Wait until each service publishes its Agent Card.
- Stop the child processes and close their log handles.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

import asyncio
import atexit
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import httpx

import settings


A2A_DIR = Path(__file__).resolve().parent


class AgentProcessManager:
    """Start, await readiness of, and stop Planner and Hotel A2A processes."""

    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.log_files: list[object] = []
        self.log_dir = Path(tempfile.mkdtemp(prefix="chapter7-a2a-"))
        atexit.register(self.stop)

    def _start_service(
        self,
        service: settings.ServiceName,
        host: str,
        port: int,
    ) -> None:
        log_path = self.log_dir / f"{service}.log"
        log_file = log_path.open("w", encoding="utf-8")
        self.log_files.append(log_file)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["A2A_SERVER_HOST"] = host
        environment[f"A2A_{service.upper()}_PORT"] = str(port)
        environment[f"A2A_{service.upper()}_URL"] = f"http://{host}:{port}/"
        environment[f"A2A_{service.upper()}_PUBLIC_URL"] = (
            f"http://{host}:{port}/"
        )
        agent_card_path = A2A_DIR / "agent_cards" / f"{service}_agent.json"
        self.processes[service] = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server",
                "--agent-card",
                str(agent_card_path),
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=A2A_DIR,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    async def _wait_until_ready(
        self,
        service: settings.ServiceName,
        base_url: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        card_url = base_url.rstrip("/") + "/.well-known/agent-card.json"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                process = self.processes[service]
                if process.poll() is not None:
                    raise RuntimeError(
                        self.failure_message(service)
                    )
                try:
                    response = await client.get(card_url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        raise TimeoutError(
            f"{service} did not become ready within {timeout_seconds:.0f}s. "
            f"Log: {self.log_dir / f'{service}.log'}"
        )

    async def start(
        self,
        *,
        host: str = settings.DEFAULT_HOST,
        ports: Mapping[str, int] | None = None,
    ) -> None:
        """Start every registered local service on available ports."""
        if self.processes:
            return

        requested_ports = dict(ports or {})
        unknown_services = set(requested_ports) - set(settings.SERVICE_NAMES)
        if unknown_services:
            raise ValueError(
                "Unknown A2A services: "
                + ", ".join(sorted(unknown_services))
            )

        selected_ports: dict[str, int] = {}
        for service in settings.SERVICE_NAMES:
            selected_port = requested_ports.get(service)
            if selected_port is None:
                selected_port = settings.find_free_port(host)
                while selected_port in selected_ports.values():
                    selected_port = settings.find_free_port(host)
            if selected_port in selected_ports.values():
                raise ValueError(
                    f"Port {selected_port} is assigned more than once"
                )
            selected_ports[service] = selected_port

        settings.configure_local_services(
            ports=selected_ports,
            host=host,
        )
        for service, port in selected_ports.items():
            self._start_service(service, host, port)

        try:
            await asyncio.gather(
                *(
                    self._wait_until_ready(
                        service,
                        settings.service_base_url(service),
                    )
                    for service in selected_ports
                )
            )
        except BaseException:
            self.stop()
            raise

    def failure_message(self, service: settings.ServiceName) -> str:
        """Return service logs when a child process exits unexpectedly."""
        log_path = self.log_dir / f"{service}.log"
        log_text = ""
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        return (
            f"{service} exited before becoming ready.\n"
            f"Log: {log_path}\n{log_text[-4000:]}"
        )

    def stop(self) -> None:
        """Terminate all child services and close their log handles."""
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.processes.clear()

        for log_file in self.log_files:
            try:
                log_file.close()
            except OSError:
                pass
        self.log_files.clear()
