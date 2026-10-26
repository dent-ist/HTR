"""FastAPI entrypoint (M0 stub)."""

from fastapi import FastAPI

# The server must only ever listen on loopback (see Makefile `serve`).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

app = FastAPI(title="HTR", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
