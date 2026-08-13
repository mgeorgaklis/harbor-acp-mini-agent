from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart

from harbor_acp_mini_agent.agent import MiniAgent, _api_model, _normalize_model


class FakeClient:
    def __init__(self) -> None:
        self.updates: list[object] = []
        self.commands: list[str] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append(update)

    async def create_terminal(self, *, command: str, args: list[str], **kwargs):
        self.commands.append(" ".join([command, *args]))
        return SimpleNamespace(terminal_id="terminal-1")

    async def wait_for_terminal_exit(self, **kwargs):
        return SimpleNamespace(exit_code=0, signal=None)

    async def terminal_output(self, **kwargs):
        return SimpleNamespace(output="created hello.txt\n", truncated=False)

    async def release_terminal(self, **kwargs):
        return None


def test_model_normalization() -> None:
    assert _normalize_model("gemini-2.5-flash-lite") == ("gemini/gemini-2.5-flash-lite")
    assert _api_model("gemini/gemini-2.5-flash-lite") == "gemini-2.5-flash-lite"
    with pytest.raises(ValueError, match="gemini"):
        _normalize_model("openai/gpt-5")


@pytest.mark.asyncio
async def test_agent_runs_shell_tool_and_returns_usage(monkeypatch) -> None:
    monkeypatch.setenv("HARBOR_ACP_REQUESTED_MODEL", "gemini/gemini-2.5-flash-lite")
    agent = MiniAgent()
    client = FakeClient()
    agent.on_connect(client)  # type: ignore[arg-type]
    session = await agent.new_session(cwd="/workspace")
    agent._generate = AsyncMock(
        side_effect=[
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "run_shell",
                                        "args": {
                                            "command": "printf 'Hello, world!' > hello.txt"
                                        },
                                    }
                                }
                            ],
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
            },
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Created hello.txt."}],
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5},
            },
        ]
    )

    response = await agent.prompt(
        session.session_id,
        [{"type": "text", "text": "Create hello.txt"}],  # type: ignore[list-item]
    )

    assert response.stop_reason == "end_turn"
    assert response.usage is not None
    assert response.usage.input_tokens == 32
    assert response.usage.output_tokens == 9
    assert client.commands == ["/bin/sh -lc printf 'Hello, world!' > hello.txt"]
    assert any(isinstance(update, ToolCallStart) for update in client.updates)
    assert any(isinstance(update, ToolCallProgress) for update in client.updates)
    assert any(isinstance(update, AgentMessageChunk) for update in client.updates)


@pytest.mark.asyncio
async def test_agent_advertises_and_updates_model(monkeypatch) -> None:
    monkeypatch.setenv("HARBOR_ACP_REQUESTED_MODEL", "gemini/gemini-2.5-flash-lite")
    agent = MiniAgent()
    session = await agent.new_session(cwd="/workspace")

    option = session.config_options[0]
    assert option.category == "model"
    assert option.current_value == "gemini/gemini-2.5-flash-lite"

    response = await agent.set_config_option(
        "model", session.session_id, "gemini/gemini-2.5-flash"
    )
    assert response.config_options[0].current_value == "gemini/gemini-2.5-flash"
