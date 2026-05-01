"""Unit tests for backend/anthropic_foundry.py translation layer.

Pure shape tests — no network, no credentials, no Anthropic SDK import.
Mocks SDK response objects with types.SimpleNamespace.
"""

import json
import os
import sys
import types

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.anthropic_foundry import (
    anthropic_response_to_openai,
    openai_messages_to_anthropic,
    openai_tools_to_anthropic,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    """Build a SimpleNamespace from kwargs."""
    return types.SimpleNamespace(**kwargs)


def _make_response(
    content_blocks,
    stop_reason="end_turn",
    input_tokens=10,
    output_tokens=20,
    cache_read=0,
    cache_creation=0,
    model="claude-sonnet-4-6",
    response_id="msg_test",
):
    """Construct a fake anthropic.types.Message-shaped object."""
    usage = _ns(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    return _ns(
        id=response_id,
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# openai_messages_to_anthropic — system handling
# ---------------------------------------------------------------------------


def test_messages_empty_input_returns_none_and_empty_list():
    system, msgs = openai_messages_to_anthropic([])
    assert system is None
    assert msgs == []


def test_messages_no_system_with_plain_user():
    system, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "Hello"},
    ])
    # No system content provided → system is None (falsy is acceptable)
    assert system is None
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_messages_multiple_leading_systems_joined_with_double_newline():
    system, _msgs = openai_messages_to_anthropic([
        {"role": "system", "content": "Block A"},
        {"role": "system", "content": "Block B"},
        {"role": "system", "content": "Block C"},
        {"role": "user", "content": "hi"},
    ], enable_cache=False)
    # enable_cache=False → plain string, joined with \n\n
    assert system == "Block A\n\nBlock B\n\nBlock C"


def test_messages_enable_cache_true_wraps_system_as_ephemeral_block():
    system, _ = openai_messages_to_anthropic([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ], enable_cache=True)
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "You are helpful."
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_messages_enable_cache_false_returns_plain_string():
    system, _ = openai_messages_to_anthropic([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ], enable_cache=False)
    assert isinstance(system, str)
    assert system == "You are helpful."


def test_messages_mid_conversation_system_folds_into_system_block_not_user():
    system, msgs = openai_messages_to_anthropic([
        {"role": "system", "content": "Initial system."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": "Mid-stream system update."},
        {"role": "user", "content": "second"},
    ], enable_cache=True)

    # Mid-stream system is appended as a new text block on the system list
    assert isinstance(system, list)
    assert len(system) == 2
    assert system[0]["text"] == "Initial system."
    assert system[1]["text"] == "Mid-stream system update."

    # No system role injected as a user message — only the two real users + assistant
    user_count = sum(1 for m in msgs if m["role"] == "user")
    assert user_count == 2
    # Verify "Mid-stream system update." does not appear in any message content
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            assert "Mid-stream system update" not in c


def test_messages_mid_system_with_no_initial_system_creates_system():
    """Mid-conversation system message with no leading system should still set system."""
    system, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "and?"},
    ], enable_cache=False)
    # System was None, mid-stream system populates it as a string
    assert system == "be terse"
    assert all(m["role"] in ("user", "assistant") for m in msgs)


# ---------------------------------------------------------------------------
# openai_messages_to_anthropic — multimodal user content
# ---------------------------------------------------------------------------


def test_messages_user_multimodal_text_image_url_and_base64():
    base64_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANS"
    system, msgs = openai_messages_to_anthropic([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at these:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "image_url", "image_url": {"url": base64_data_uri}},
            ],
        },
    ])
    assert len(msgs) == 1
    blocks = msgs[0]["content"]
    assert isinstance(blocks, list)
    assert len(blocks) == 3
    assert blocks[0] == {"type": "text", "text": "Look at these:"}
    # Plain URL → url-source image
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["type"] == "url"
    assert blocks[1]["source"]["url"] == "https://example.com/cat.png"
    # Data URI → base64 source with media_type extracted
    assert blocks[2]["type"] == "image"
    assert blocks[2]["source"]["type"] == "base64"
    assert blocks[2]["source"]["media_type"] == "image/png"
    assert blocks[2]["source"]["data"] == "iVBORw0KGgoAAAANS"


# ---------------------------------------------------------------------------
# openai_messages_to_anthropic — assistant tool_calls
# ---------------------------------------------------------------------------


def test_messages_assistant_with_tool_calls_emits_tool_use_block():
    args = {"location": "Jakarta", "unit": "celsius"}
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps(args),
                },
            }],
        },
    ])
    assistant_msg = msgs[1]
    assert assistant_msg["role"] == "assistant"
    blocks = assistant_msg["content"]
    assert isinstance(blocks, list)
    # Should have a text block (preamble) and a tool_use block
    assert any(b.get("type") == "text" for b in blocks)
    tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
    assert len(tool_blocks) == 1
    tu = tool_blocks[0]
    assert tu["id"] == "call_abc"
    assert tu["name"] == "get_weather"
    # arguments string was JSON-decoded into input dict
    assert tu["input"] == args
    assert isinstance(tu["input"], dict)


def test_messages_assistant_tool_call_with_empty_arguments():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_x",
                "type": "function",
                "function": {"name": "list_files", "arguments": ""},
            }],
        },
    ])
    blocks = msgs[1]["content"]
    tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["input"] == {}


# ---------------------------------------------------------------------------
# openai_messages_to_anthropic — tool message coalescing
# ---------------------------------------------------------------------------


def test_messages_single_tool_message_coalesces_into_user_with_tool_result():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 28C"},
    ])
    # Last message is a synthesized user with tool_result
    last = msgs[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    assert len(last["content"]) == 1
    block = last["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "Sunny, 28C"


def test_messages_multiple_consecutive_tool_messages_merge_into_one_user():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "many things"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "a", "arguments": "{}"}},
                {"id": "c2", "type": "function",
                 "function": {"name": "b", "arguments": "{}"}},
                {"id": "c3", "type": "function",
                 "function": {"name": "c", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result A"},
        {"role": "tool", "tool_call_id": "c2", "content": "result B"},
        {"role": "tool", "tool_call_id": "c3", "content": "result C"},
    ])
    # Last user message coalesces all three tool results into a single user message
    user_msgs_after_assistant = [m for m in msgs if m["role"] == "user"]
    last = user_msgs_after_assistant[-1]
    assert isinstance(last["content"], list)
    tool_results = [b for b in last["content"] if b.get("type") == "tool_result"]
    assert len(tool_results) == 3
    assert [b["tool_use_id"] for b in tool_results] == ["c1", "c2", "c3"]
    assert [b["content"] for b in tool_results] == ["result A", "result B", "result C"]


def test_messages_tool_result_with_list_content_passed_through():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "a", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": "block 1"},
                        {"type": "text", "text": "block 2"}],
        },
    ])
    last = msgs[-1]
    block = last["content"][0]
    assert block["type"] == "tool_result"
    assert isinstance(block["content"], list)
    assert len(block["content"]) == 2


# ---------------------------------------------------------------------------
# openai_messages_to_anthropic — consecutive same-role merging
# ---------------------------------------------------------------------------


def test_messages_consecutive_user_strings_merged():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ])
    # Two strings → joined with \n\n
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "first\n\nsecond"


def test_messages_consecutive_user_lists_merged():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ])
    assert len(msgs) == 1
    assert isinstance(msgs[0]["content"], list)
    assert len(msgs[0]["content"]) == 2


def test_messages_consecutive_string_then_list_merged_to_list():
    _, msgs = openai_messages_to_anthropic([
        {"role": "user", "content": "preamble"},
        {"role": "user", "content": [{"type": "text", "text": "follow"}]},
    ])
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert isinstance(content, list)
    # Preamble was promoted to a text block, then list appended
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "preamble"
    assert content[-1]["text"] == "follow"


def test_messages_consecutive_unsupported_content_raises_value_error():
    """When merge encounters a content type that's neither str nor list,
    a clear ValueError surfaces so the caller can fail fast."""
    from backend.anthropic_foundry import _merge_consecutive_same_role
    with pytest.raises(ValueError) as exc_info:
        _merge_consecutive_same_role([
            {"role": "user", "content": {"unexpected": "dict"}},
            {"role": "user", "content": "trailing string"},
        ])
    msg = str(exc_info.value)
    assert "user" in msg
    assert "Cannot merge" in msg or "merge" in msg.lower()


# ---------------------------------------------------------------------------
# openai_tools_to_anthropic
# ---------------------------------------------------------------------------


def test_tools_empty_input_returns_empty_list():
    assert openai_tools_to_anthropic([]) == []


def test_tools_none_input_returns_empty_list():
    # Implementation chose [] for falsy input; assert that consistently
    assert openai_tools_to_anthropic(None) == []


def test_tools_single_tool_strips_wrapper_and_renames_parameters():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": schema,
        },
    }]
    out = openai_tools_to_anthropic(tools)
    assert len(out) == 1
    assert out[0]["name"] == "get_weather"
    assert out[0]["description"] == "Look up weather"
    assert out[0]["input_schema"] == schema
    # Wrapper keys should not survive
    assert "function" not in out[0]
    assert "type" not in out[0]
    assert "parameters" not in out[0]


def test_tools_multiple_tools_preserve_order():
    tools = [
        {"type": "function", "function": {
            "name": "alpha", "description": "A",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "beta", "description": "B",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "gamma", "description": "G",
            "parameters": {"type": "object", "properties": {}}}},
    ]
    out = openai_tools_to_anthropic(tools)
    assert [t["name"] for t in out] == ["alpha", "beta", "gamma"]
    assert [t["description"] for t in out] == ["A", "B", "G"]


# ---------------------------------------------------------------------------
# anthropic_response_to_openai
# ---------------------------------------------------------------------------


def test_response_text_only():
    resp = _make_response(
        content_blocks=[_ns(type="text", text="Hello world")],
        stop_reason="end_turn",
        input_tokens=5,
        output_tokens=3,
    )
    out = anthropic_response_to_openai(resp, duration_ms=42, model_name="claude-sonnet-4-6")
    msg = out["response"]["choices"][0]["message"]
    assert msg["content"] == "Hello world"
    assert "tool_calls" not in msg or not msg.get("tool_calls")
    assert "reasoning_content" not in msg
    assert out["success"] is True
    assert out["duration_ms"] == 42


def test_response_with_thinking_populates_reasoning_content():
    resp = _make_response(content_blocks=[
        _ns(type="thinking", thinking="Let me reason..."),
        _ns(type="text", text="Final answer"),
    ])
    out = anthropic_response_to_openai(resp, duration_ms=10, model_name="m")
    msg = out["response"]["choices"][0]["message"]
    assert msg["reasoning_content"] == "Let me reason..."
    assert msg["content"] == "Final answer"


def test_response_redacted_thinking_renders_as_marker():
    resp = _make_response(content_blocks=[
        _ns(type="redacted_thinking"),
        _ns(type="text", text="Here you go"),
    ])
    out = anthropic_response_to_openai(resp, duration_ms=1, model_name="m")
    msg = out["response"]["choices"][0]["message"]
    assert msg["reasoning_content"] == "[redacted thinking]"


def test_response_with_tool_use_blocks_populates_tool_calls():
    args1 = {"a": 1}
    args2 = {"b": 2}
    resp = _make_response(
        content_blocks=[
            _ns(type="tool_use", id="tu_1", name="first_tool", input=args1),
            _ns(type="tool_use", id="tu_2", name="second_tool", input=args2),
        ],
        stop_reason="tool_use",
    )
    out = anthropic_response_to_openai(resp, duration_ms=5, model_name="m")
    msg = out["response"]["choices"][0]["message"]
    assert msg["tool_calls"] is not None
    assert len(msg["tool_calls"]) == 2
    # Order preserved
    assert msg["tool_calls"][0]["id"] == "tu_1"
    assert msg["tool_calls"][0]["function"]["name"] == "first_tool"
    assert msg["tool_calls"][1]["id"] == "tu_2"
    assert msg["tool_calls"][1]["function"]["name"] == "second_tool"
    # Arguments are JSON strings
    assert isinstance(msg["tool_calls"][0]["function"]["arguments"], str)
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == args1
    assert json.loads(msg["tool_calls"][1]["function"]["arguments"]) == args2
    # finish_reason should be "tool_calls" via stop_reason mapping
    assert out["response"]["choices"][0]["finish_reason"] == "tool_calls"


def test_response_mixed_text_thinking_and_tool_use():
    resp = _make_response(content_blocks=[
        _ns(type="thinking", thinking="thinking through"),
        _ns(type="text", text="visible text"),
        _ns(type="tool_use", id="t_1", name="do_thing", input={"x": 1}),
    ])
    out = anthropic_response_to_openai(resp, duration_ms=1, model_name="m")
    msg = out["response"]["choices"][0]["message"]
    assert msg["content"] == "visible text"
    assert msg["reasoning_content"] == "thinking through"
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "do_thing"


@pytest.mark.parametrize("stop_reason,expected_finish", [
    ("end_turn", "stop"),
    ("max_tokens", "length"),
    ("tool_use", "tool_calls"),
    ("stop_sequence", "stop"),
    ("something_unknown", "stop"),
    (None, "stop"),
    ("", "stop"),
])
def test_response_stop_reason_mapping(stop_reason, expected_finish):
    resp = _make_response(
        content_blocks=[_ns(type="text", text="x")],
        stop_reason=stop_reason,
    )
    out = anthropic_response_to_openai(resp, duration_ms=1, model_name="m")
    assert out["response"]["choices"][0]["finish_reason"] == expected_finish


def test_response_token_accounting_sums_input_and_cache_buckets():
    resp = _make_response(
        content_blocks=[_ns(type="text", text="hi")],
        input_tokens=100,
        output_tokens=50,
        cache_read=200,
        cache_creation=30,
    )
    out = anthropic_response_to_openai(resp, duration_ms=1, model_name="m")
    # prompt_tokens = input + cache_read + cache_creation
    assert out["prompt_tokens"] == 100 + 200 + 30
    assert out["completion_tokens"] == 50
    assert out["total_tokens"] == 100 + 200 + 30 + 50
    # Mirrored on the response.usage dict
    usage = out["response"]["usage"]
    assert usage["prompt_tokens"] == 330
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 380
    # Raw breakdown preserved for future dashboards
    assert usage["cache_read_input_tokens"] == 200
    assert usage["cache_creation_input_tokens"] == 30


def test_response_zero_usage_safe():
    resp = _make_response(
        content_blocks=[_ns(type="text", text="hi")],
        input_tokens=0,
        output_tokens=0,
    )
    out = anthropic_response_to_openai(resp, duration_ms=1, model_name="m")
    assert out["prompt_tokens"] == 0
    assert out["completion_tokens"] == 0
    assert out["total_tokens"] == 0


def test_response_echoes_model_name_when_response_missing_model():
    resp = _ns(
        id="msg_x",
        model=None,
        content=[_ns(type="text", text="hi")],
        stop_reason="end_turn",
        usage=_ns(input_tokens=1, output_tokens=1,
                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )
    out = anthropic_response_to_openai(resp, duration_ms=1, model_name="claude-haiku-4-5")
    assert out["response"]["model"] == "claude-haiku-4-5"
