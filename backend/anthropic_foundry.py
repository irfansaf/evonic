"""Translation layer between OpenAI-shape requests/responses and Anthropic's native API.

Pure-Python module — no Flask, SQLite, or SDK calls. Only converts data shapes
so the canonical LLMClient can keep its OpenAI-flavoured contract while talking
to Azure AI Foundry's Anthropic-native Claude endpoint.

Functions:
    openai_messages_to_anthropic: split system blocks from messages, convert
        roles/content/tool calls to Anthropic block structure, and coalesce
        consecutive tool results into a single user message.
    openai_tools_to_anthropic: strip the OpenAI ``{"type":"function","function":...}``
        wrapper and rename ``parameters`` -> ``input_schema``.
    anthropic_response_to_openai: rebuild the OpenAI ``chat.completion`` shape
        the rest of the codebase expects, including ``tool_calls``,
        ``reasoning_content``, and ``finish_reason``.

Designed so streaming support can be layered on later (the message/tool
converters do not assume non-streaming).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.normalizer import normalize_llm_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_URI_RE = re.compile(r"^data:(?P<media>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _normalize_text(text: Any) -> str:
    """Apply LLMClient's quote normalization, treating non-strings as-is."""
    if not isinstance(text, str):
        return text  # type: ignore[return-value]
    return normalize_llm_text(text)


def _convert_user_content_list(content: List[Any]) -> List[Dict[str, Any]]:
    """Convert an OpenAI multimodal user content list into Anthropic blocks."""
    blocks: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            # Plain string fragment in a list — treat as text
            if isinstance(item, str):
                blocks.append({"type": "text", "text": _normalize_text(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            blocks.append({"type": "text", "text": _normalize_text(item.get("text", ""))})
        elif item_type == "image_url":
            url_field = item.get("image_url")
            url = url_field.get("url") if isinstance(url_field, dict) else url_field
            if not isinstance(url, str) or not url:
                continue
            data_match = _DATA_URI_RE.match(url)
            if data_match:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": data_match.group("media"),
                        "data": data_match.group("data"),
                    },
                })
            else:
                blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        elif item_type == "image":
            # Pass through Anthropic-shape image blocks unchanged
            blocks.append(item)
        else:
            # Unknown block type — best-effort: keep text if present, else drop
            text_field = item.get("text")
            if isinstance(text_field, str):
                blocks.append({"type": "text", "text": _normalize_text(text_field)})
    return blocks


def _flush_tool_results(buffer: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert a list of buffered ``role:tool`` messages into a single user message
    containing tool_result blocks. Returns None when the buffer is empty.
    """
    if not buffer:
        return None
    blocks: List[Dict[str, Any]] = []
    for tool_msg in buffer:
        tool_use_id = tool_msg.get("tool_call_id") or tool_msg.get("id") or ""
        tool_content = tool_msg.get("content", "")
        if isinstance(tool_content, list):
            # Already block-shaped — pass through
            tool_result_content: Union[str, List[Dict[str, Any]]] = tool_content
        else:
            tool_result_content = _normalize_text(tool_content) if tool_content is not None else ""
        blocks.append({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": tool_result_content,
        })
    return {"role": "user", "content": blocks}


def _merge_consecutive_same_role(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic rejects consecutive same-role messages. Merge their content arrays.

    String + list mixes raise ValueError so the caller can surface a clear,
    fail-fast error to the user rather than letting the SDK 422.
    """
    if not messages:
        return messages
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1].get("role") == msg.get("role"):
            prev = merged[-1]
            prev_content = prev.get("content")
            cur_content = msg.get("content")
            if isinstance(prev_content, list) and isinstance(cur_content, list):
                prev["content"] = prev_content + cur_content
            elif isinstance(prev_content, str) and isinstance(cur_content, str):
                prev["content"] = prev_content + "\n\n" + cur_content
            elif isinstance(prev_content, list) and isinstance(cur_content, str):
                prev["content"] = prev_content + [
                    {"type": "text", "text": cur_content}
                ]
            elif isinstance(prev_content, str) and isinstance(cur_content, list):
                prev["content"] = [{"type": "text", "text": prev_content}] + cur_content
            else:
                raise ValueError(
                    f"Cannot merge consecutive {msg.get('role')!r} messages: "
                    f"unsupported content types ({type(prev_content).__name__}, "
                    f"{type(cur_content).__name__})"
                )
        else:
            merged.append(dict(msg))
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def openai_messages_to_anthropic(
    messages: List[Dict[str, Any]],
    enable_cache: bool = True,
) -> Tuple[Optional[Union[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    """Translate OpenAI-shape messages into ``(system, anthropic_messages)``.

    Args:
        messages: OpenAI ``chat.completions`` messages (system/user/assistant/tool).
        enable_cache: When True the joined system prompt is wrapped as an
            ephemeral-cache text block. When False it is returned as a plain
            string (or None when no system content was provided).

    Returns:
        Tuple ``(system, anthropic_messages)``:
        - ``system``: cache-wrapped block list, plain string, or None.
        - ``anthropic_messages``: list of Anthropic-shape ``{"role","content"}`` dicts.

    Raises:
        ValueError: when post-conversion messages cannot be merged due to
            mixed content types — surfaces as fail-fast misconfiguration.
    """
    if not messages:
        return None, []

    # Extract leading system messages
    system_texts: List[str] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        sys_content = messages[idx].get("content", "")
        if isinstance(sys_content, list):
            # Flatten any block-shape system content into plain text
            parts = [
                blk.get("text", "")
                for blk in sys_content
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            sys_content = "\n".join(p for p in parts if p)
        if sys_content:
            system_texts.append(_normalize_text(sys_content))
        idx += 1

    if system_texts:
        joined_system = "\n\n".join(system_texts)
        if enable_cache:
            system_out: Optional[Union[str, List[Dict[str, Any]]]] = [{
                "type": "text",
                "text": joined_system,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_out = joined_system
    else:
        system_out = None

    # Walk remaining messages
    anthropic_messages: List[Dict[str, Any]] = []
    tool_buffer: List[Dict[str, Any]] = []

    def flush() -> None:
        flushed = _flush_tool_results(tool_buffer)
        if flushed is not None:
            anthropic_messages.append(flushed)
        tool_buffer.clear()

    for msg in messages[idx:]:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "tool":
            tool_buffer.append(msg)
            continue

        # Hit a non-tool message — flush any buffered tool results first
        if tool_buffer:
            flush()

        if role == "system":
            # Mid-conversation system message — fold into the running system block
            sys_text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                sys_text = "\n".join(
                    blk.get("text", "")
                    for blk in content
                    if isinstance(blk, dict) and blk.get("type") == "text"
                )
            if sys_text:
                normalized = _normalize_text(sys_text)
                if isinstance(system_out, list):
                    # Append a new (uncached) text block
                    system_out.append({"type": "text", "text": normalized})
                elif isinstance(system_out, str):
                    system_out = system_out + "\n\n" + normalized
                else:
                    system_out = normalized
            continue

        if role == "user":
            if isinstance(content, list):
                anthropic_messages.append({
                    "role": "user",
                    "content": _convert_user_content_list(content),
                })
            else:
                anthropic_messages.append({
                    "role": "user",
                    "content": _normalize_text(content or ""),
                })
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                blocks: List[Dict[str, Any]] = []
                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": _normalize_text(content)})
                elif isinstance(content, list):
                    blocks.extend(_convert_user_content_list(content))
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments", "{}") or "{}"
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            # Preserve the broken string so the assistant turn still
                            # round-trips; Anthropic will surface the issue clearly.
                            parsed_args = {"_raw_arguments": raw_args}
                    else:
                        parsed_args = raw_args
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": parsed_args,
                    })
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                if isinstance(content, list):
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": _convert_user_content_list(content),
                    })
                else:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": _normalize_text(content or ""),
                    })
            continue

        # Unknown role — skip silently
        continue

    # Flush any trailing buffered tool messages
    if tool_buffer:
        flush()

    # Coalesce consecutive same-role messages (ValueError on incompatible mixes)
    anthropic_messages = _merge_consecutive_same_role(anthropic_messages)

    return system_out, anthropic_messages


def openai_tools_to_anthropic(
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Convert an OpenAI tool list into Anthropic's ``{"name","description","input_schema"}`` shape."""
    if not tools:
        return []
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Already Anthropic-shape — pass through
        if "input_schema" in tool and "name" in tool:
            converted.append(tool)
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        out: Dict[str, Any] = {"name": name}
        description = fn.get("description")
        if description:
            out["description"] = description
        params = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        out["input_schema"] = params
        converted.append(out)
    return converted


def _stop_reason_to_finish_reason(stop_reason: Optional[str]) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }
    return mapping.get(stop_reason or "", "stop")


def anthropic_response_to_openai(
    response: Any,
    duration_ms: int,
    model_name: str,
) -> Dict[str, Any]:
    """Convert an Anthropic ``Message`` SDK object into LLMClient's success-dict shape.

    Args:
        response: ``anthropic.types.Message`` returned from ``messages.create``.
        duration_ms: Round-trip latency captured by the caller.
        model_name: Model identifier to echo in the response payload.

    Returns:
        Dict matching the success branch of ``LLMClient.chat_completion``:
        ``{"response": {...openai-shape...}, "duration_ms", "prompt_tokens",
        "completion_tokens", "total_tokens", "success": True}``.
    """
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in getattr(response, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif block_type == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif block_type == "redacted_thinking":
            # Opaque thinking block — surface a marker so reasoning isn't silently dropped
            thinking_parts.append("[redacted thinking]")
        elif block_type == "tool_use":
            tool_input = getattr(block, "input", {}) or {}
            try:
                args_json = json.dumps(tool_input, ensure_ascii=False)
            except (TypeError, ValueError):
                args_json = json.dumps(str(tool_input))
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": args_json,
                },
            })

    content_text = "".join(text_parts)
    # Apply quote-normalization so downstream consumers behave identically to OpenAI path
    content_text = _normalize_text(content_text) if content_text else content_text
    thinking_text = "\n".join(p for p in thinking_parts if p)

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": content_text,
    }
    if thinking_text:
        message["reasoning_content"] = thinking_text
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = _stop_reason_to_finish_reason(getattr(response, "stop_reason", None))

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
    # Anthropic reports cached/non-cached input separately; sum to a single OpenAI-like total
    total_prompt_tokens = prompt_tokens + cache_read + cache_creation
    total_tokens = total_prompt_tokens + completion_tokens

    openai_shape = {
        "id": getattr(response, "id", ""),
        "object": "chat.completion",
        "model": getattr(response, "model", model_name) or model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }

    return {
        "response": openai_shape,
        "duration_ms": duration_ms,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "success": True,
    }
