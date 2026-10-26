"""The `suggest` job: ask the local LLM about abbreviation-like tokens and store the answers as reviewable suggestions.

Runs one token at a time (cancellable between tokens). Results are stored as they arrive, so a cancelled or failed
job keeps what it already produced. Nothing here touches a transcription.
"""

from __future__ import annotations

import logging

from ..abbreviation_llm import LlmClient, LlmConfigError, LlmError
from ..errors import JobError
from ..logging_setup import log_event
from ..services import suggestions as suggestions_svc
from .handlers import JobContext

log = logging.getLogger(__name__)

DISABLED_HINT = (
    "Set HTR_LLM_ENABLED=1 (and run a local model server such as Ollama) before starting the app; see the README, "
    "section 'Local LLM abbreviation suggestions'."
)


def make_client(settings) -> LlmClient:
    try:
        return LlmClient(settings.llm_url, settings.llm_model, settings.llm_api, settings.llm_timeout)
    except LlmConfigError as e:
        hint = "Fix HTR_LLM_URL / HTR_LLM_API / HTR_LLM_MODEL and restart the app."
        raise JobError(str(e), "llm_config", hint) from e


def suggest_abbreviations(ctx: JobContext) -> dict:
    """payload: {tokens?, max_tokens?, context_lines?}. Result: {tokens, asked, suggestions_added, model}."""
    s = ctx.settings
    if not s.llm_enabled:
        raise JobError("Local-LLM suggestions are turned off.", "llm_disabled", DISABLED_HINT)
    client = make_client(s)
    limit = ctx.payload.get("max_tokens") or suggestions_svc.DEFAULT_MAX_TOKENS
    context_lines = ctx.payload.get("context_lines")
    context_lines = s.llm_context_lines if context_lines is None else context_lines
    todo = suggestions_svc.find_candidates(ctx.conn, limit, ctx.payload.get("tokens"), context_lines)
    summary = {"tokens": len(todo), "asked": 0, "suggestions_added": 0, "model": client.model}
    ctx.info(summary)
    for i, item in enumerate(todo):
        ctx.check_cancelled(summary)
        try:
            candidates = client.suggest(item.token, item.line, item.before, item.after)
        except LlmError as e:
            log_event(log, logging.WARNING, "llm request failed", code=e.code, asked=summary["asked"])
            raise JobError(e.message, e.code, e.hint) from e
        summary["asked"] += 1
        summary["suggestions_added"] += suggestions_svc.store(ctx.conn, item, candidates, client.model)
        ctx.progress((i + 1) / len(todo))
        ctx.info(summary)
    log_event(log, logging.INFO, "suggestions generated", asked=summary["asked"], added=summary["suggestions_added"])
    return summary
