"""A hosted copy of the page (https) talking to the engine on this computer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

SITE = "https://my-cut.example.app"


@pytest.fixture()
def client(settings, monkeypatch):
    from possession_cut import config
    from possession_cut.api.main import create_app

    monkeypatch.setenv("CORS_ORIGINS", SITE)
    config.get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    config.get_settings.cache_clear()


def preflight(client: TestClient, origin: str, method: str = "PUT", private: bool = True):
    headers = {"Origin": origin, "Access-Control-Request-Method": method, "Access-Control-Request-Headers": "content-type"}
    if private:
        headers["Access-Control-Request-Private-Network"] = "true"
    return client.options("/api/uploads/" + "0" * 32 + "?offset=0", headers=headers)


def test_listed_site_may_call_the_local_engine(client):
    # Chrome sends this before every request from a public page to a local address. It
    # must come back 2xx with the permission, or the page cannot reach the engine at all.
    r = preflight(client, SITE)
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == SITE
    assert r.headers["access-control-allow-private-network"] == "true"
    assert "PUT" in r.headers["access-control-allow-methods"]
    for method in ("GET", "POST", "PATCH", "DELETE"):
        assert preflight(client, SITE, method).status_code == 200

    health = client.get("/api/health", headers={"Origin": SITE})
    assert health.status_code == 200 and health.headers["access-control-allow-origin"] == SITE


def test_the_local_dev_page_is_always_allowed(client):
    assert preflight(client, "http://localhost:5173", private=False).status_code == 200


def test_unlisted_site_is_refused(client):
    r = preflight(client, "https://somewhere-else.example")
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers
    # the browser enforces it on real requests by the missing header
    assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": "https://somewhere-else.example"}).headers
