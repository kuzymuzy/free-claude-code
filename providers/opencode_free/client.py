"""OpenCode Free provider — no API key, UUID session rotation on 429."""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from core.anthropic import SSEBuilder, map_stop_reason
from providers.base import BaseProvider, ProviderConfig
from providers.defaults import OPENCODE_DEFAULT_BASE
from providers.model_listing import extract_openai_model_ids
from providers.opencode.request import build_request_body

_POOL_SIZE = 8
_COOLDOWN_SKIPS = 3  # skip a session N times after a 429


class _SessionPool:
    """Manage a pool of session UUIDs, rotating on 429 responses."""

    def __init__(self, size: int) -> None:
        self._pool: deque[str] = deque(str(uuid.uuid4()) for _ in range(size))
        self._skip: dict[str, int] = {}

    def current(self) -> str:
        """Return the first session id that is not in cooldown."""
        for _ in range(len(self._pool)):
            sid = self._pool[0]
            remaining = self._skip.get(sid, 0)
            if remaining <= 0:
                return sid
            self._skip[sid] = remaining - 1
            self._pool.rotate(-1)
        return self._pool[0]

    def rotate(self, bad: str) -> None:
        """Mark *bad* for cooldown and rotate the pool."""
        logger.warning("opencode_free: 429 → rotating away from session {}…", bad[:8])
        self._skip[bad] = _COOLDOWN_SKIPS
        if self._pool[0] == bad:
            self._pool.rotate(-1)


_pool = _SessionPool(_POOL_SIZE)


async def _openai_sse_chunks(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed JSON chunks from an OpenAI-compatible SSE stream."""
    async for raw_line in resp.aiter_lines():
        if not raw_line:
            continue
        if raw_line.startswith("data: [DONE]"):
            return
        if raw_line.startswith("data: "):
            line = raw_line.removeprefix("data: ")
        elif raw_line.startswith("[DONE]"):
            return
        else:
            continue
        if not line or not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


class OpenCodeFreeProvider(BaseProvider):
    """OpenCode Free — anonymous access with UUID session rotation on 429."""

    def __init__(
        self, config: ProviderConfig, provider_name: str = "OPENCODE_FREE"
    ) -> None:
        super().__init__(config)
        self._provider_name = provider_name
        self._base_url = (config.base_url or OPENCODE_DEFAULT_BASE).rstrip("/")
        self._timeout = config.http_read_timeout
        self._proxy = config.proxy

    async def cleanup(self) -> None:
        """No-op: no persistent connections to release."""

    async def list_model_ids(self) -> frozenset[str]:
        """Fetch model list from the upstream /models endpoint."""
        url = f"{self._base_url}/models"
        transport = httpx.AsyncHTTPTransport(proxy=self._proxy) if self._proxy else None
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=transport
        ) as client:
            try:
                resp = await client.get(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "opencode/1.0",
                    },
                )
                resp.raise_for_status()
                return extract_openai_model_ids(
                    resp.json(), provider_name=self._provider_name
                )
            except Exception:
                return frozenset()

    def _build_headers(self, session_id: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-opencode-session": session_id,
            "x-session-affinity": session_id,
            "User-Agent": "opencode/1.0",
        }

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        body = build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
        body["stream"] = True
        url = f"{self._base_url}/chat/completions"

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        sse = SSEBuilder(
            message_id,
            request.model,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        for attempt in range(_POOL_SIZE):
            session_id = _pool.current()
            headers = self._build_headers(session_id)

            logger.debug(
                "opencode_free: attempt {}/{} session {}… model={}",
                attempt + 1,
                _POOL_SIZE,
                session_id[:8],
                body.get("model"),
            )

            transport = (
                httpx.AsyncHTTPTransport(proxy=self._proxy) if self._proxy else None
            )
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=transport,
            ) as client, client.stream(
                "POST", url, headers=headers, json=body
            ) as resp:
                if resp.status_code == 429:
                    _pool.rotate(session_id)
                    continue
                resp.raise_for_status()

                yield sse.message_start()

                finish_reason: str | None = None
                started_thinking = False
                started_text = False

                async for chunk in _openai_sse_chunks(resp):
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Reasoning content → Anthropic thinking block
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        if started_text:
                            yield sse.stop_text_block()
                            started_text = False
                        if not started_thinking:
                            yield sse.start_thinking_block()
                            started_thinking = True
                        yield sse.emit_thinking_delta(reasoning)

                    # Text content
                    content = delta.get("content")
                    if content:
                        if started_thinking:
                            yield sse.stop_thinking_block()
                            started_thinking = False
                        if not started_text:
                            yield sse.start_text_block()
                            started_text = True
                        yield sse.emit_text_delta(content)

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                # Close any open content blocks
                if started_thinking:
                    yield sse.stop_thinking_block()
                if started_text:
                    yield sse.stop_text_block()

                yield sse.message_delta(map_stop_reason(finish_reason), None)
                yield sse.message_stop()
                return

        raise RuntimeError(
            "opencode_free: all sessions are rate-limited, try again later"
        )
