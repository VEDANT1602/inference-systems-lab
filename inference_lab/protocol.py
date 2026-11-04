"""A small OpenAI-compatible streaming client, written without an SDK dependency."""
from __future__ import annotations

import codecs
from dataclasses import dataclass, field
import hashlib
import http.client
import json
import itertools
import math
import socket
import ssl
import threading
import time
from typing import Iterable, Iterator
from urllib.parse import urlsplit


class ProtocolError(ValueError):
    pass


def sse_events(chunks: Iterable[bytes], max_bytes: int = 65536) -> Iterator[str]:
    """Decode UTF-8 incrementally; event boundaries are blank lines, not TCP reads."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer, data, consumed = "", [], 0
    for chunk, final in itertools.chain(((c, False) for c in chunks), [(b"", True)]):
        consumed += len(chunk)
        if consumed > max_bytes:
            raise ProtocolError("response_too_large")
        buffer += decoder.decode(chunk, final=final)
        while "\n" in buffer or "\r" in buffer:
            boundary = min(i for i in (buffer.find("\n"), buffer.find("\r")) if i >= 0)
            if buffer[boundary] == "\r" and boundary == len(buffer) - 1 and not final:
                break
            width = 2 if buffer[boundary:boundary+2] == "\r\n" else 1
            line, buffer = buffer[:boundary], buffer[boundary+width:]
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
            # Comments, ids, retry and event names do not alter data payloads.
    if buffer or data:
        raise ProtocolError("unterminated_sse_event")


@dataclass(frozen=True)
class Endpoint:
    alias: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Endpoint must be an HTTP(S) base URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Credentials, queries and fragments are not allowed in endpoint URLs")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Cleartext HTTP is restricted to loopback endpoints")
        if not self.alias or not self.model:
            raise ValueError("Endpoint alias and model must be nonempty")

    @property
    def remote(self) -> bool:
        return urlsplit(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}


def quality(text: str, expected_label: str, finish_reason: str | None) -> tuple[bool, str, str | None]:
    if finish_reason != "stop":
        return False, "generation_did_not_stop_normally", None
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return False, "invalid_json", None
    if not isinstance(value, dict) or set(value) != {"label"}:
        return False, "schema_mismatch", None
    if not isinstance(value["label"], str) or value["label"] not in {"billing", "technical", "account"}:
        return False, "unknown_label", None
    return value["label"] == expected_label, "correct" if value["label"] == expected_label else "wrong_label", value["label"]


def stream_chat(endpoint: Endpoint, case: dict, *, timeout_s: float = 5.0,
                execute_remote: bool = False) -> dict:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if endpoint.remote and not execute_remote:
        raise ValueError("Remote requests require explicit --execute-remote")
    parsed = urlsplit(endpoint.base_url)
    path = parsed.path.rstrip("/") + "/chat/completions"
    conn_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    kwargs = {"timeout": timeout_s}
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    conn = conn_type(parsed.hostname, parsed.port, **kwargs)
    body = json.dumps({"model": endpoint.model, "messages": case["messages"],
                       "temperature": 0, "max_tokens": case.get("max_tokens", 32),
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if endpoint.api_key:
        headers["Authorization"] = "Bearer " + endpoint.api_key
    started = time.perf_counter()
    result = {"case_id": case["case_id"], "endpoint_alias": endpoint.alias,
              "model": endpoint.model, "transport_success": False,
              "quality_pass": False, "error": None, "finish_reason": None,
              "usage": None, "content_chunks": 0, "content_chunk_gap_ms": [],
              "ttft_proxy_ms": None, "inter_token_latency_ms": None,
              "timing_boundary": "worker_start_to_first_nonempty_content_delivery"}
    text, stamps, done, response = "", [], False, None
    deadline_expired, watchdog = threading.Event(), None
    try:
        conn.connect()
        transport_socket = conn.sock
        def remaining_timeout():
            remaining = timeout_s - (time.perf_counter() - started)
            if deadline_expired.is_set() or remaining <= 0:
                raise TimeoutError("request_deadline")
            transport_socket.settimeout(remaining)
        remaining_timeout()
        def expire_connected_socket():
            # Socket timeouts are idle timeouts inside getresponse(). A peer that
            # trickles header bytes must not reset the request's remaining budget.
            deadline_expired.set()
            try:
                transport_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        watchdog = threading.Timer(max(0, timeout_s - (time.perf_counter() - started)),
                                   expire_connected_socket)
        watchdog.daemon = True
        watchdog.start()
        conn.request("POST", path, body=body, headers=headers)
        remaining_timeout()
        response = conn.getresponse()
        remaining_timeout()
        result["http_status"] = response.status
        if response.status != 200:
            result["error"] = "http_" + str(response.status)
            return result
        if not response.getheader("Content-Type", "").lower().startswith("text/event-stream"):
            raise ProtocolError("unexpected_content_type")

        def chunks():
            while True:
                # Retain the public socket handle even if HTTPConnection detaches it.
                remaining_timeout()
                chunk = response.read1(4096)
                if not chunk:
                    break
                yield chunk

        for event in sse_events(chunks()):
            now = time.perf_counter()
            if event == "[DONE]":
                done = True
                break
            data = json.loads(event)
            if not isinstance(data, dict) or "error" in data:
                raise ProtocolError("provider_error_event")
            if data.get("usage") is not None:
                usage = data["usage"]
                if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                                      for k in ("prompt_tokens", "completion_tokens")):
                    raise ProtocolError("invalid_usage")
                result["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens")}
            choices = data.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise ProtocolError("unsupported_choices")
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ProtocolError("unsupported_choice_index")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ProtocolError("invalid_delta")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ProtocolError("invalid_content")
                if content:
                    text += content
                    stamps.append(now)
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]
        if not done or result["finish_reason"] is None or not stamps:
            raise ProtocolError("incomplete_stream")
        classification = quality(text, case["expected_label"], result["finish_reason"])
        if deadline_expired.is_set() or time.perf_counter() - started >= timeout_s:
            raise TimeoutError("request_deadline")
        result["transport_success"] = True
        result["quality_pass"], result["quality_reason"], result["received_label"] = classification
    except (TimeoutError, socket.timeout):
        result["error"] = "timeout"
    except (ProtocolError, ValueError, UnicodeError, TypeError, RecursionError):
        result["error"] = "timeout" if deadline_expired.is_set() else "protocol_error"
    except (OSError, http.client.HTTPException):
        result["error"] = "timeout" if deadline_expired.is_set() else "transport_error"
    finally:
        if watchdog is not None:
            watchdog.cancel()
        result["e2e_ms"] = (time.perf_counter() - started) * 1000
        result["content_chunks"] = len(stamps)
        if stamps:
            result["ttft_proxy_ms"] = (stamps[0] - started) * 1000
        result["content_chunk_gap_ms"] = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
        result["response_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        if response is not None:
            response.close()
        conn.close()
    return result
