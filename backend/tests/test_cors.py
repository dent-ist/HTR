import pytest

ALLOWED = ["http://localhost:5173", "http://127.0.0.1:5173"]
DENIED = [
    "http://localhost:3000",
    "http://evil.example",
    "https://localhost:5173",
    "http://localhost:5173.evil.example",
    "null",
]


@pytest.mark.parametrize("origin", ALLOWED)
def test_allowed_origins(client, origin):
    r = client.get("/health", headers={"Origin": origin})
    assert r.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-credentials" not in r.headers


@pytest.mark.parametrize("origin", ALLOWED)
def test_preflight_allowed(client, origin):
    r = client.options(
        "/lines/1",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
    assert "PATCH" in r.headers["access-control-allow-methods"]


@pytest.mark.parametrize("origin", DENIED)
def test_other_origins_get_no_cors_headers(client, origin):
    r = client.get("/health", headers={"Origin": origin})
    assert "access-control-allow-origin" not in r.headers
    pre = client.options("/lines/1", headers={"Origin": origin, "Access-Control-Request-Method": "PATCH"})
    assert pre.status_code == 400 and "access-control-allow-origin" not in pre.headers


def test_no_wildcard_anywhere(client):
    from app.config import CORS_ORIGINS

    assert CORS_ORIGINS == ALLOWED
    r = client.get("/health", headers={"Origin": ALLOWED[0]})
    assert r.headers["access-control-allow-origin"] != "*"
    pre = client.options("/lines/1", headers={"Origin": ALLOWED[0], "Access-Control-Request-Method": "DELETE"})
    assert pre.headers["access-control-allow-headers"] != "*" and "*" not in pre.headers["access-control-allow-methods"]
