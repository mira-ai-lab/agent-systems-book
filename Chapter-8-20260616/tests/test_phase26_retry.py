"""Phase 26.8：router-client 重试 / 超时 / 错误类型。"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent / "packages" / "router-client"


def test_router_client_http_policy_module():
    assert (PKG_ROOT / "src" / "http.ts").is_file()
    http_src = (PKG_ROOT / "src" / "http.ts").read_text(encoding="utf-8")
    types_src = (PKG_ROOT / "src" / "types.ts").read_text(encoding="utf-8")
    index_src = (PKG_ROOT / "src" / "index.ts").read_text(encoding="utf-8")
    client_src = (PKG_ROOT / "src" / "client.ts").read_text(encoding="utf-8")

    assert "fetchWithPolicy" in http_src
    assert "isRetryableStatus" in http_src
    assert "RouterClientTimeoutError" in types_src
    assert "RouterClientNetworkError" in types_src
    assert "FetchPolicy" in types_src
    assert "fetchWithPolicy" in client_src
    assert "RouterClientTimeoutError" in index_src
    assert (PKG_ROOT / "tests" / "http.test.mjs").is_file()
