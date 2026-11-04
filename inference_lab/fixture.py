"""A known-answer loopback server for protocol tests; it is not a model emulator."""
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time

from .protocol import Endpoint


class FixtureServer(AbstractContextManager):
    def __init__(self, cases, mode="correct", chunk_delay=0.001):
        answers = {case["case_id"]: case["expected_label"] for case in cases}
        self.requests = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                owner.requests += 1
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                messages = body.get("messages", [])
                text = "\n".join(m.get("content", "") for m in messages)
                matched = re.search(r"case_id=([a-z0-9-]+)", text)
                key = matched.group(1) if matched else None
                if self.path != "/v1/chat/completions" or key not in answers:
                    self.send_error(404)
                    return
                if mode == "http_error":
                    self.send_response(429)
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    return
                if mode == "trickle_headers":
                    self.close_connection = True
                    try:
                        for byte in b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n":
                            self.connection.sendall(bytes([byte]))
                            time.sleep(.01)
                    except OSError:
                        pass
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                if mode == "timeout":
                    time.sleep(.25)
                label = answers[key]
                if mode == "wrong_label":
                    label = "billing" if label != "billing" else "account"
                answer = json.dumps({"label": label}, separators=(",", ":")) if mode != "invalid_json" else "not-json"
                if mode == "deep_answer":
                    answer = "[" * 1100 + "]" * 1100
                events = [": heartbeat café\r\n\r\n",
                          "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\r\n\r\n"]
                content_size = 1100 if mode == "deep_answer" else 5
                for start in range(0, len(answer), content_size):
                    events.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": answer[start:start+content_size]}, "finish_reason": None}]}) + "\r\n\r\n")
                events.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\r\n\r\n")
                if mode != "no_usage":
                    usage = {"prompt_tokens": 64, "completion_tokens": True if mode == "bad_usage" else 6}
                    events.append("data: " + json.dumps({"choices": [], "usage": usage}) + "\r\n\r\n")
                if mode != "interrupted":
                    events.append("data: [DONE]\r\n\r\n")
                if mode == "deep_event":
                    events = ["data: " + "[" * 1100 + "]" * 1100 + "\n\n"]
                try:
                    for event in events:
                        encoded = event.encode("utf-8")
                        if mode == "trickle_body":
                            for byte in encoded:
                                self.connection.sendall(bytes([byte]))
                                time.sleep(.01)
                            continue
                        # Fragment UTF-8 and SSE delimiters across network writes.
                        midpoint = max(1, len(encoded) // 2)
                        self.wfile.write(encoded[:midpoint])
                        self.wfile.flush()
                        self.wfile.write(encoded[midpoint:])
                        self.wfile.flush()
                        time.sleep(chunk_delay)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.endpoint = Endpoint("loopback-protocol-fixture", "http://127.0.0.1:" + str(self.server.server_port) + "/v1", "known-answer-fixture")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def original_cases():
    shared = ("Classify a fictional customer request. Return exactly one JSON object with only label. "
              "Use billing for invoices, refunds and payments; technical for product errors and connectivity; "
              "account for sign-in, credentials and account access. Do not perform an account action. ")
    examples = [("b01", "billing", "The invoice charged my subscription twice."),
                ("b02", "billing", "Where is the refund for my cancelled plan?"),
                ("b03", "billing", "My card payment was rejected."),
                ("b04", "billing", "Can I get a receipt for last month's payment?"),
                ("t01", "technical", "The export button produces an application error."),
                ("t02", "technical", "The service disconnects while I upload a file."),
                ("t03", "technical", "My desktop client crashes on startup."),
                ("t04", "technical", "The integration returns a timeout."),
                ("a01", "account", "I forgot my password and cannot sign in."),
                ("a02", "account", "How do I change my account email?"),
                ("a03", "account", "My two-factor sign-in code is not accepted."),
                ("a04", "account", "I need to restore access to my disabled account.")]
    return [{"case_id": key, "messages": [{"role": "system", "content": shared},
                                            {"role": "user", "content": "case_id=" + key + "\n" + text}],
             "expected_label": label, "prefix_group": "helpdesk-policy-v1", "max_tokens": 32}
            for key, label, text in examples]
