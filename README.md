# Harbor ACP Mini Agent

A deliberately small coding agent that demonstrates Harbor's source-backed
[Agent Client Protocol](https://agentclientprotocol.com/) integration.

The agent sends the task instruction to Gemini, exposes one `run_shell` tool,
and loops until the model returns a final response. Shell commands are executed
through the ACP client, so Harbor retains control of the task environment and
records tool events in the resulting trajectory.

This is an example and smoke-test agent, not a production replacement for
Mini-SWE-Agent.

## Repository contract

Harbor expects the repository to contain:

- `harbor-agent.json`, describing the ACP entrypoint.
- `pyproject.toml` and a committed `uv.lock`.
- An ACP agent that communicates over standard input and output.

The source-backed Harbor configuration is:

```yaml
agents:
  - name: acp
    model_name: gemini/gemini-3.5-flash-lite
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
    kwargs:
      source:
        repo_url: https://github.com/mgeorgaklis/harbor-acp-mini-agent.git
        ref: <full-commit-sha>
        manifest_path: harbor-agent.json
```

Use a full commit SHA for reproducible evaluations. Harbor fetches and validates
the repository in the controller, uploads the verified tree into the task
sandbox, runs `uv sync --frozen`, and executes the ACP entrypoint only inside the
sandbox.

## Local checks

```bash
uv lock --check
uv run --with pytest==8.4.2 pytest
```

Set `GEMINI_API_KEY` and connect any ACP client to
`uv run harbor-acp-mini-agent` for a live run.
