"""Trusted-side opaque signature roundtrip; never invents or bypasses signatures.

OpenCode still owns the loop/tools/history. Both original wire and restored API
wire are journaled. Only provider-issued opaque metadata is restored by tool ID.
"""
import copy
import json
from .contract import ParityError, canonical


class GeminiTransport:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    def __init__(self, key):
        self.key = key
        self.calls = {}
        self.streams = {}

    @property
    def auth_headers(self):
        return {'authorization': 'Bearer ' + self.key}

    def prepare(self, body):
        obj = json.loads(body)
        if obj.get("model") != "gemini-3.8-flash" or obj.get("temperature") != 1:
            raise ParityError("API model/temperature differs from contract")
        for message in obj.get("messages", []):
            for call in message.get("tool_calls", []):
                stored = self.calls.get(call.get("id"))
                if stored and stored.get("extra_content"):
                    if call.get("extra_content") not in (None, stored["extra_content"]):
                        raise ParityError("Provider signature changed")
                    if call.get("function") != stored["function"]:
                        # JSON argument serialization can change without changing values.
                        a, b = call.get("function", {}), stored["function"]
                        if a.get("name") != b.get("name") or json.loads(a.get("arguments", "{}")) != json.loads(b.get("arguments", "{}")):
                            raise ParityError("Signed tool call semantics changed")
                    call["extra_content"] = copy.deepcopy(stored["extra_content"])
        return canonical(obj)

    def feed(self, request_id, chunk):
        state = self.streams.setdefault(request_id, {"buffer": b"", "calls": {}, "ids": {}, "done": False})
        state["buffer"] += chunk
        outgoing = []
        while b"\n" in state["buffer"]:
            line, state["buffer"] = state["buffer"].split(b"\n", 1)
            if not line.startswith(b"data:"):
                outgoing.append(line + b'\n')
                continue
            if line[5:].strip() == b"[DONE]":
                state['done'] = True
                outgoing.append(line + b'\n')
                continue
            obj = json.loads(line[5:])
            for choice in obj.get("choices", []):
                for delta in choice.get("delta", {}).get("tool_calls", []):
                    # Gemini's actual compatible SSE emits complete ID-bearing
                    # calls without OpenAI's required index. Reconstruct only
                    # this transport index; ambiguous fragments fail closed.
                    if 'index' not in delta:
                        if not delta.get('id') or not delta.get('function', {}).get('name'):
                            raise ParityError('Cannot index an unidentified API tool fragment')
                        json.loads(delta['function']['arguments'])
                        delta['index'] = state['ids'].setdefault(delta['id'], len(state['ids']))
                    elif delta.get('id'):
                        state['ids'][delta['id']] = delta['index']
                    call = state["calls"].setdefault(delta["index"], {"function": {"name": "", "arguments": ""}})
                    if delta.get("id"):
                        call["id"] = delta["id"]
                    for key in ("name", "arguments"):
                        call["function"][key] += delta.get("function", {}).get(key) or ""
                    if "extra_content" in delta:
                        call["extra_content"] = delta["extra_content"]
                    if "id" in call:
                        self.calls[call["id"]] = call
                # The provider also reports `stop` after functionCall parts.
                # OpenCode requires tool_calls to continue the native loop.
                if state['calls'] and choice.get('finish_reason') == 'stop':
                    choice['finish_reason'] = 'tool_calls'
            outgoing.append(b'data: ' + canonical(obj) + b'\n')
        return b''.join(outgoing)

    def finish(self, request_id):
        state = self.streams.pop(request_id, None)
        if state and (state['buffer'].strip() or not state['done']):
            raise ParityError('Incomplete API SSE stream')
