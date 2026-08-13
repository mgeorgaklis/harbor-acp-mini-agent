from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    start_tool_call,
    text_block,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
    SseMcpServer,
    TextContentBlock,
    Usage,
)

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"
MAX_STEPS = 20
MAX_TOOL_OUTPUT_CHARS = 20_000
_MODEL_PATTERN = re.compile(r"^(?:gemini/)?gemini-[A-Za-z0-9._-]+$")

SYSTEM_PROMPT = """You are a compact software-engineering agent running in a task sandbox.
Use run_shell to inspect and modify the workspace. Work directly in the current directory.
Continue until the requested change is complete, verify it when practical, then return a
short final summary. Do not only describe commands: execute them."""

SHELL_TOOL = {
    "functionDeclarations": [
        {
            "name": "run_shell",
            "description": "Run a shell command in the task workspace and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "POSIX shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        }
    ]
}


@dataclass
class Session:
    cwd: str
    model: str
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0

    def add(self, metadata: Any) -> None:
        if not isinstance(metadata, dict):
            return
        self.input_tokens += _non_negative_int(metadata.get("promptTokenCount"))
        self.output_tokens += _non_negative_int(metadata.get("candidatesTokenCount"))
        self.thought_tokens += _non_negative_int(metadata.get("thoughtsTokenCount"))
        self.cached_tokens += _non_negative_int(metadata.get("cachedContentTokenCount"))

    def to_acp(self) -> Usage:
        return Usage(
            total_tokens=self.input_tokens + self.output_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            thought_tokens=self.thought_tokens or None,
            cached_read_tokens=self.cached_tokens or None,
        )


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _normalize_model(value: str) -> str:
    model = value.strip()
    if not _MODEL_PATTERN.fullmatch(model):
        raise ValueError("Harbor ACP Mini Agent supports gemini/* models only")
    return model if model.startswith("gemini/") else f"gemini/{model}"


def _api_model(model: str) -> str:
    return _normalize_model(model).split("/", 1)[1]


def _prompt_text(prompt: list[Any]) -> str:
    chunks: list[str] = []
    for block in prompt:
        text = (
            block.get("text")
            if isinstance(block, dict)
            else getattr(block, "text", None)
        )
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


class MiniAgent(Agent):
    _conn: Client

    def __init__(self) -> None:
        requested_model = os.environ.get("HARBOR_ACP_REQUESTED_MODEL", DEFAULT_MODEL)
        self._default_model = _normalize_model(requested_model)
        self._sessions: dict[str, Session] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    def _model_option(self, model: str) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            current_value=model,
            options=[SessionConfigSelectOption(value=model, name=model)],
            id="model",
            name="Model",
            description="Gemini model used by this agent",
            category="model",
            type="select",
        )

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(
                name="harbor-acp-mini-agent",
                title="Harbor ACP Mini Agent",
                version="0.1.2",
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid4().hex
        self._sessions[session_id] = Session(cwd=cwd, model=self._default_model)
        return NewSessionResponse(
            session_id=session_id,
            config_options=[self._model_option(self._default_model)],
        )

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse:
        session = self._sessions[session_id]
        if config_id != "model" or not isinstance(value, str):
            raise ValueError("Only the string model configuration is supported")
        session.model = _normalize_model(value)
        return SetSessionConfigOptionResponse(
            config_options=[self._model_option(session.model)]
        )

    async def prompt(
        self,
        session_id: str,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        **kwargs: Any,
    ) -> PromptResponse:
        session = self._sessions[session_id]
        session.cancelled.clear()
        instruction = _prompt_text(prompt)
        if not instruction:
            await self._send_message(session_id, "No text instruction was provided.")
            return PromptResponse(stop_reason="refusal")

        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": instruction}]}
        ]
        usage = TokenUsage()

        for step in range(MAX_STEPS):
            if session.cancelled.is_set():
                return PromptResponse(stop_reason="cancelled", usage=usage.to_acp())

            response = await self._generate(session.model, contents)
            usage.add(response.get("usageMetadata"))
            content = self._response_content(response)
            parts = content.get("parts")
            if not isinstance(parts, list):
                raise RuntimeError("Gemini response did not contain content parts")
            contents.append(content)

            function_calls = [
                part["functionCall"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)
            ]
            for part in parts:
                if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                    continue
                if part.get("thought"):
                    await self._conn.session_update(
                        session_id, update_agent_thought(text_block(part["text"]))
                    )

            if not function_calls:
                final_text = "\n".join(
                    part["text"]
                    for part in parts
                    if isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                    and not part.get("thought")
                ).strip()
                await self._send_message(session_id, final_text or "Task complete.")
                return PromptResponse(stop_reason="end_turn", usage=usage.to_acp())

            tool_parts: list[dict[str, Any]] = []
            for index, call in enumerate(function_calls):
                name = call.get("name")
                arguments = call.get("args")
                if name != "run_shell" or not isinstance(arguments, dict):
                    result = {"error": f"Unsupported function: {name!r}"}
                else:
                    command = arguments.get("command")
                    if not isinstance(command, str) or not command.strip():
                        result = {"error": "run_shell requires a non-empty command"}
                    else:
                        result = await self._run_shell(
                            session_id=session_id,
                            cwd=session.cwd,
                            command=command,
                            step=step,
                            index=index,
                        )
                tool_parts.append(
                    {
                        "functionResponse": {
                            "name": str(name or "unknown"),
                            "response": result,
                        }
                    }
                )
            contents.append({"role": "user", "parts": tool_parts})

        await self._send_message(
            session_id, "Stopped after reaching the tool-step limit."
        )
        return PromptResponse(stop_reason="max_turn_requests", usage=usage.to_acp())

    async def _send_message(self, session_id: str, message: str) -> None:
        await self._conn.session_update(
            session_id, update_agent_message(text_block(message))
        )

    async def _run_shell(
        self,
        *,
        session_id: str,
        cwd: str,
        command: str,
        step: int,
        index: int,
    ) -> dict[str, Any]:
        tool_call_id = f"shell-{step}-{index}-{uuid4().hex[:8]}"
        await self._conn.session_update(
            session_id,
            start_tool_call(
                tool_call_id,
                "Run shell command",
                kind="execute",
                status="in_progress",
                raw_input={"command": command},
            ),
        )
        terminal = await self._conn.create_terminal(
            session_id=session_id,
            command="/bin/sh",
            args=["-lc", command],
            cwd=cwd,
            output_byte_limit=MAX_TOOL_OUTPUT_CHARS,
        )
        try:
            exit_status = await self._conn.wait_for_terminal_exit(
                session_id=session_id,
                terminal_id=terminal.terminal_id,
            )
            output = await self._conn.terminal_output(
                session_id=session_id,
                terminal_id=terminal.terminal_id,
            )
        finally:
            await self._conn.release_terminal(
                session_id=session_id,
                terminal_id=terminal.terminal_id,
            )

        result = {
            "exit_code": exit_status.exit_code,
            "signal": exit_status.signal,
            "output": output.output[-MAX_TOOL_OUTPUT_CHARS:],
            "truncated": output.truncated,
        }
        await self._conn.session_update(
            session_id,
            update_tool_call(
                tool_call_id,
                status="completed" if exit_status.exit_code == 0 else "failed",
                raw_output=result,
            ),
        )
        return result

    async def _generate(
        self, model: str, contents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "tools": [SHELL_TOOL],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        }
        return await asyncio.to_thread(
            self._post_json,
            _api_model(model),
            api_key,
            payload,
        )

    @staticmethod
    def _post_json(model: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    value = json.loads(response.read(4 * 1024 * 1024))
                if not isinstance(value, dict):
                    raise RuntimeError("Gemini returned a non-object response")
                return value
            except urllib.error.HTTPError as error:
                detail = error.read(16_384).decode("utf-8", "replace")
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(
                        f"Gemini request failed with HTTP {error.code}: {detail}"
                    ) from None
            except urllib.error.URLError as error:
                if attempt == 2:
                    raise RuntimeError(
                        f"Gemini request failed: {error.reason}"
                    ) from None
            time.sleep(2**attempt)
        raise RuntimeError("Gemini request failed after retries")

    @staticmethod
    def _response_content(response: dict[str, Any]) -> dict[str, Any]:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {response}")
        content = candidates[0].get("content")
        if not isinstance(content, dict):
            raise RuntimeError("Gemini candidate did not contain content")
        return content

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        if session := self._sessions.get(session_id):
            session.cancelled.set()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    await run_agent(MiniAgent())


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
