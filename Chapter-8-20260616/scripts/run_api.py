"""启动 HTTP API 服务。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8780"))
    print(
        f"Multi-Agent Platform API → http://{host}:{port}\n"
        f"  health: GET /health   domains: GET /v1/domains\n"
        f"  chat: POST /v1/chat  (仅 query 即可，profile 默认 auto；domain 可省略)"
    )
    uvicorn.run("services.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
