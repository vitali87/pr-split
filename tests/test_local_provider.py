from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pr_split.config import Settings
from pr_split.constants import (
    LOCAL_BASE_URL,
    LOCAL_MAX_CONTEXT_TOKENS,
    LOCAL_MAX_OUTPUT_TOKENS,
    PartitionStrategy,
    Provider,
)
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import LLMError
from pr_split.planner.client import _call_llm, _count_tokens, _json_from_text, plan_split
from pr_split.planner.prompts import SPLIT_TOOL_NAME

_DIFF = """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1,2 @@
+x = 1
+y = 2
"""

_GROUPS = [
    {
        "id": "pr-1",
        "title": "Add a.py",
        "description": "Adds a.py",
        "depends_on": [],
        "assignments": [
            {"file_path": "a.py", "assignment_type": "whole_file", "hunk_indices": [0]}
        ],
        "estimated_loc": 2,
    }
]


def _tool_call_reply(arguments: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": SPLIT_TOOL_NAME, "arguments": arguments},
            }
        ],
    }


class _FakeServer:
    """An OpenAI-compatible /v1/chat/completions endpoint, as Ollama or llama.cpp serve."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.message: dict[str, object] = _tool_call_reply(json.dumps({"groups": _GROUPS}))
        self.finish_reason = "tool_calls"
        self.status = 200
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                fake.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        **json.loads(self.rfile.read(length)),
                    }
                )
                if fake.status != 200:
                    body = json.dumps({"error": {"message": "model not found"}}).encode()
                else:
                    body = json.dumps(
                        {
                            "id": "chatcmpl-1",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "m",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": fake.message,
                                    "finish_reason": fake.finish_reason,
                                }
                            ],
                        }
                    ).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> _FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[_FakeServer]:
    with _FakeServer() as fake:
        yield fake


def _local_settings(url: str, **overrides: object) -> Settings:
    return Settings(
        provider=Provider.LOCAL,
        partition_strategy=PartitionStrategy.LLM,
        model=overrides.pop("model", "qwen2.5-coder:14b"),
        local_base_url=url,
        **overrides,  # type: ignore[arg-type]
    )


class TestLocalSettings:
    def test_needs_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(provider=Provider.LOCAL, partition_strategy="llm", model="m")
        assert settings.local_base_url == LOCAL_BASE_URL
        assert settings.max_context_tokens == LOCAL_MAX_CONTEXT_TOKENS
        assert settings.max_output_tokens == LOCAL_MAX_OUTPUT_TOKENS

    def test_requires_a_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PR_SPLIT_MODEL", raising=False)
        with pytest.raises(ValueError, match="PR_SPLIT_MODEL must be set"):
            Settings(provider=Provider.LOCAL, partition_strategy="llm")

    def test_model_not_needed_without_the_llm_backend(self) -> None:
        settings = Settings(provider=Provider.LOCAL, partition_strategy="graph")
        assert settings.model == ""

    def test_output_budget_must_fit_the_window(self) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            Settings(
                provider=Provider.LOCAL,
                partition_strategy="llm",
                model="m",
                local_context_tokens=4096,
                local_max_output_tokens=4096,
            )

    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PR_SPLIT_PROVIDER", "local")
        monkeypatch.setenv("PR_SPLIT_MODEL", "llama3.1:8b")
        monkeypatch.setenv("PR_SPLIT_LOCAL_BASE_URL", "http://gpu-box:8000/v1")
        monkeypatch.setenv("PR_SPLIT_LOCAL_CONTEXT_TOKENS", "131072")
        monkeypatch.setenv("PR_SPLIT_LOCAL_MAX_OUTPUT_TOKENS", "16384")
        monkeypatch.setenv("PR_SPLIT_LOCAL_API_KEY", "secret")
        settings = Settings(partition_strategy="llm")
        assert settings.provider is Provider.LOCAL
        assert settings.model == "llama3.1:8b"
        assert settings.local_base_url == "http://gpu-box:8000/v1"
        assert settings.max_context_tokens == 131072
        assert settings.max_output_tokens == 16384
        assert settings.api_key == "secret"

    def test_cloud_providers_keep_their_output_budget(self) -> None:
        settings = Settings(provider=Provider.OPENAI, partition_strategy="graph")
        assert settings.max_output_tokens == 128_000


class TestCallLocal:
    def test_forced_tool_call_is_the_plan(self, server: _FakeServer) -> None:
        result = _call_llm("the system", "the diff", settings=_local_settings(server.url))

        assert result["groups"] == _GROUPS
        request = server.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["model"] == "qwen2.5-coder:14b"
        assert request["temperature"] == 0
        assert request["max_tokens"] == LOCAL_MAX_OUTPUT_TOKENS
        assert request["messages"] == [
            {"role": "system", "content": "the system"},
            {"role": "user", "content": "the diff"},
        ]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": SPLIT_TOOL_NAME},
        }
        assert request["tools"][0]["function"]["name"] == SPLIT_TOOL_NAME  # type: ignore[index]

    def test_plan_in_message_text_is_accepted(self, server: _FakeServer) -> None:
        server.message = {
            "role": "assistant",
            "content": "Here is the plan:\n```json\n" + json.dumps({"groups": _GROUPS}) + "\n```",
        }
        server.finish_reason = "stop"
        result = _call_llm("s", "u", settings=_local_settings(server.url))
        assert result["groups"] == _GROUPS

    def test_call_written_as_text_is_unwrapped(self, server: _FakeServer) -> None:
        server.message = {
            "role": "assistant",
            "content": json.dumps({"name": SPLIT_TOOL_NAME, "arguments": {"groups": _GROUPS}}),
        }
        server.finish_reason = "stop"
        result = _call_llm("s", "u", settings=_local_settings(server.url))
        assert result["groups"] == _GROUPS

    def test_prose_reply_is_an_llm_error(self, server: _FakeServer) -> None:
        server.message = {"role": "assistant", "content": "I cannot split this."}
        server.finish_reason = "stop"
        with pytest.raises(LLMError, match="no tool call or JSON plan"):
            _call_llm("s", "u", settings=_local_settings(server.url))

    def test_cut_off_reply_is_truncated(self, server: _FakeServer) -> None:
        server.message = _tool_call_reply('{"groups": [{"id": "pr-1"')
        server.finish_reason = "length"
        with pytest.raises(LLMError, match="cut off"):
            _call_llm("s", "u", settings=_local_settings(server.url))

    def test_bad_tool_arguments_are_an_llm_error(self, server: _FakeServer) -> None:
        server.message = _tool_call_reply("not json")
        with pytest.raises(LLMError, match="failed to parse tool arguments"):
            _call_llm("s", "u", settings=_local_settings(server.url))

    def test_server_error_is_an_llm_error(self, server: _FakeServer) -> None:
        server.status = 404
        with pytest.raises(LLMError, match="model not found"):
            _call_llm("s", "u", settings=_local_settings(server.url))

    def test_api_key_is_sent_when_set(self, server: _FakeServer) -> None:
        _call_llm("s", "u", settings=_local_settings(server.url, local_api_key="secret"))
        headers = server.requests[0]["headers"]
        assert headers["Authorization"] == "Bearer secret"  # type: ignore[index]

    def test_unreachable_server_names_the_url(self) -> None:
        with _FakeServer() as fake:
            url = fake.url
        settings = _local_settings(url)
        with pytest.raises(LLMError, match=f"Cannot reach the local LLM server at {url}"):
            _call_llm("s", "u", settings=settings)

    def test_tokens_are_estimated_locally(self) -> None:
        assert _count_tokens("system", "user text", settings=_local_settings(LOCAL_BASE_URL)) > 0


class TestPlanSplitLocal:
    def test_end_to_end_plan(self, server: _FakeServer) -> None:
        groups = plan_split(parse_diff(_DIFF), _local_settings(server.url))
        assert [g.id for g in groups] == ["pr-1"]
        assert groups[0].estimated_loc == 2

    def test_large_diff_is_chunked_within_a_small_window(self, server: _FakeServer) -> None:
        files = "".join(
            f"diff --git a/f{i}.py b/f{i}.py\nnew file mode 100644\n--- /dev/null\n"
            f"+++ b/f{i}.py\n@@ -0,0 +1,40 @@\n"
            + "".join(f"+value_{i}_{n} = {n} * 12345678\n" for n in range(40))
            for i in range(30)
        )
        group = {
            **_GROUPS[0],
            "assignments": [{**_GROUPS[0]["assignments"][0], "file_path": "f0.py"}],
        }  # type: ignore[dict-item]
        server.message = _tool_call_reply(json.dumps({"groups": [group]}))
        settings = _local_settings(
            server.url, local_context_tokens=16_384, local_max_output_tokens=2_048
        )

        groups = plan_split(parse_diff(files), settings)

        assert len(server.requests) > 1
        for request in server.requests:
            assert request["max_tokens"] == 2_048
        # Hunks the model left out are still placed in its group.
        assert sum(len(a.hunk_indices) for g in groups for a in g.assignments) == 30


class TestJsonFromText:
    @pytest.mark.parametrize("text", ["", "no json here", "{broken", "[1, 2]"])
    def test_rejects_non_objects(self, text: str) -> None:
        assert _json_from_text(text) is None

    def test_bare_object(self) -> None:
        assert _json_from_text('{"groups": []}') == {"groups": []}
