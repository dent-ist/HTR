"""A fake local LLM server on 127.0.0.1 that mimics the Ollama chat API and the OpenAI-compatible chat endpoint
(llama.cpp's llama-server). It records every request so tests can check exactly what would have been sent."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Reply = Callable[[str], str]


def default_reply(user_prompt: str) -> str:
    token = user_prompt.split('"')[1] if '"' in user_prompt else "?"
    return json.dumps(
        {
            "candidates": [
                {"expansion": f"{token} expanded", "rationale": "fits the sentence"},
                {"expansion": f"{token} alternative", "rationale": "also possible"},
            ]
        }
    )


class FakeLlm:
    def __init__(self, api: str = "ollama", reply: Reply = default_reply, delay: float = 0.0, status: int = 200):
        self.api = api
        self.reply = reply
        self.delay = delay
        self.status = status
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output quiet
                pass

            def _send(self, code: int, payload: dict):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append({"method": "GET", "path": self.path, "body": None})
                self._send(outer.status, {"models": []} if outer.api == "ollama" else {"data": []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append({"method": "POST", "path": self.path, "body": body})
                if outer.delay:
                    threading.Event().wait(outer.delay)
                if outer.status != 200:
                    return self._send(outer.status, {"error": "simulated"})
                user = body["messages"][-1]["content"]
                text = outer.reply(user)
                if outer.api == "ollama" and self.path == "/api/chat":
                    self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": text}})
                elif outer.api == "openai" and self.path == "/v1/chat/completions":
                    self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})
                else:
                    self._send(404, {"error": "unknown endpoint"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def prompts(self) -> list[str]:
        return [r["body"]["messages"][-1]["content"] for r in self.requests if r["method"] == "POST"]

    def __enter__(self) -> FakeLlm:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
