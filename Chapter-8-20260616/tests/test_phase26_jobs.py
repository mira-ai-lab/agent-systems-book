"""Phase 26.7：router-client jobs API。"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent / "packages" / "router-client"


def test_router_client_jobs_api_in_source():
    client_src = (PKG_ROOT / "src" / "client.ts").read_text(encoding="utf-8")
    types_src = (PKG_ROOT / "src" / "types.ts").read_text(encoding="utf-8")
    integration = (PKG_ROOT / "tests" / "integration.test.mjs").read_text(encoding="utf-8")

    assert "buildJobRequestBody" in client_src
    assert "async submitJob(" in client_src
    assert "async getJob(" in client_src
    assert "JobSubmitResponse" in types_src
    assert "JobRecord" in types_src
    assert "submitJob() and getJob()" in integration
