"""OpenCode Free provider — no API key, UUID session rotation on 429."""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from core.anthropic import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_stop_reason,
)
from providers.base import BaseProvider, ProviderConfig
from providers.defaults import OPENCODE_DEFAULT_BASE
from providers.model_listing import extract_openai_model_ids
from providers.opencode.request import build_request_body
from providers.transports.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
    iter_heuristic_tool_use_sse,
)

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
            async with (
                httpx.AsyncClient(
                    timeout=self._timeout,
                    transport=transport,
                ) as client,
                client.stream("POST", url, headers=headers, json=body) as resp,
            ):
                if resp.status_code == 429:
                    _pool.rotate(session_id)
                    continue
                resp.raise_for_status()

                yield sse.message_start()

                think_parser = ThinkTagParser()
                heuristic_parser = HeuristicToolParser()
                tool_assembler = OpenAIToolCallAssembler()
                finish_reason: str | None = None

                async for chunk in _openai_sse_chunks(resp):
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    # Reasoning content → Anthropic thinking block
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        for event in sse.ensure_thinking_block():
                            yield event
                        yield sse.emit_thinking_delta(reasoning)

                    # Text content — feed through ThinkTagParser → HeuristicToolParser
                    content = delta.get("content")
                    if content:
                        for part in think_parser.feed(content):
                            if part.type == ContentType.THINKING:
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(part.content)
                            else:
                                (
                                    filtered_text,
                                    detected_tools,
                                ) = heuristic_parser.feed(part.content)
                                if filtered_text:
                                    for event in sse.ensure_text_block():
                                        yield event
                                    yield sse.emit_text_delta(filtered_text)
                                for tool_use in detected_tools:
                                    for event in iter_heuristic_tool_use_sse(
                                        sse, tool_use
                                    ):
                                        yield event

                    # Structured tool calls from OpenAI SSE format
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for event in sse.close_content_blocks():
                            yield event
                        for tc in tool_calls:
                            tc_info = {
                                "index": tc.get("index", 0),
                                "id": tc.get("id"),
                                "function": {
                                    "name": tc.get("function", {}).get("name"),
                                    "arguments": tc.get("function", {}).get(
                                        "arguments", ""
                                    ),
                                },
                            }
                            for event in tool_assembler.process_tool_call(tc_info, sse):
                                yield event

                # Flush remaining content from parsers
                remaining = think_parser.flush()
                if remaining:
                    if remaining.type == ContentType.THINKING:
                        for event in sse.ensure_thinking_block():
                            yield event
                        yield sse.emit_thinking_delta(remaining.content)
                    elif remaining.type == ContentType.TEXT:
                        for event in sse.ensure_text_block():
                            yield event
                        yield sse.emit_text_delta(remaining.content)

                for tool_use in heuristic_parser.flush():
                    for event in iter_heuristic_tool_use_sse(sse, tool_use):
                        yield event

                # Close all open blocks
                for event in sse.close_all_blocks():
                    yield event

                output_tokens = sse.estimate_output_tokens()
                yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
                yield sse.message_stop()
                return

        raise RuntimeError(
            "opencode_free: all sessions are rate-limited, try again later"
        )
