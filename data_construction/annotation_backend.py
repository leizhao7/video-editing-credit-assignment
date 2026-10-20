"""Non-interactive annotation CLI adapters; no provider SDK or network calls on import."""
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from .release_config import annotation_backend, annotation_cli, env_text, require_annotation_model
except ImportError:
    from release_config import annotation_backend, annotation_cli, env_text, require_annotation_model


def _object(value):
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class AnnotationCommand:
    backend: str
    model: str
    argv: list[str]


def build_annotation_command(prompt, task_dir, *, tools, effort, mcp_config=""):
    """Build argv, never a shell command. Each CLI retains its native auth/config."""
    backend, cli, model = annotation_backend(), annotation_cli(), require_annotation_model()
    task_dir = str(Path(task_dir).resolve())
    requested_effort = env_text("VIDEO_EDITING_ANNOTATION_EFFORT")
    if mcp_config and backend != "claude":
        raise ValueError("VIDEO_EDITING_MCP_CONFIG is Claude-only; unset it and configure MCP "
                         "in the selected CLI's native configuration")
    if backend == "claude":
        argv = [cli, "-p", prompt, "--model", model, "--effort", requested_effort or effort,
                "--allowedTools", tools, "--disallowedTools", "Agent,Task",
                "--permission-mode", "acceptEdits", "--add-dir", task_dir,
                "--output-format", "stream-json", "--verbose"]
        if mcp_config:
            argv += ["--mcp-config", str(Path(mcp_config).expanduser().resolve()), "--strict-mcp-config"]
    elif backend == "codex":
        argv = [cli, "exec", "--model", model, "--json", "--color", "never",
                "--sandbox", "workspace-write", "--skip-git-repo-check", "--cd", task_dir,
                "-c", 'approval_policy="never"', "--disable", "multi_agent"]
        if requested_effort:
            argv += ["-c", "model_reasoning_effort=" + json.dumps(requested_effort)]
        argv += ["--", prompt]
    else:
        argv = [cli, "run", "--model", model, "--format", "json", "--dir", task_dir]
        if requested_effort:
            argv += ["--variant", requested_effort]
        argv += ["--", prompt]
    return AnnotationCommand(backend, model, argv)


class AnnotationEvents:
    """Normalize progress, terminal failures, and reported cost from CLI JSONL.

    Tool failures can be recovered by the agent. Only session/turn failures fail
    the annotation; successful file validation is still required by each caller.
    Costs are provider-reported, never inferred from tokens or absent fields.
    """

    def __init__(self, backend):
        self.backend = backend
        self.cost = None
        self.rate_limit = {}
        self.errors = []
        self.completed = False
        self._seen = set()
        self._steps = set()
        self._pending_error = ""

    def _tool(self, identifier, name, arguments):
        if identifier and identifier in self._seen:
            return []
        if identifier:
            self._seen.add(identifier)
        return [{"name": name, "input": arguments if isinstance(arguments, dict) else {}}]

    def feed(self, line):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            return []
        if not isinstance(event, dict):
            return []
        kind = event.get("type")
        if self.backend == "claude":
            if kind == "assistant":
                result = []
                content = _object(event.get("message")).get("content") or []
                for part in content if isinstance(content, list) else []:
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        result += self._tool(part.get("id"), part.get("name", "tool"), part.get("input"))
                return result
            if kind == "rate_limit_event":
                self.rate_limit = _object(event.get("rate_limit_info"))
            if kind == "result":
                self.completed = not event.get("is_error", False)
                self.cost = event.get("total_cost_usd")
                if not self.completed:
                    self.errors.append(str(event.get("errors") or event.get("result") or "Claude run failed"))
        elif self.backend == "codex":
            if kind in {"item.started", "item.completed"}:
                item = _object(event.get("item"))
                item_type = item.get("type")
                if item_type == "command_execution":
                    return self._tool(item.get("id"), "Bash", {"command": item.get("command", "")})
                if item_type in {"file_change", "mcp_tool_call", "web_search"}:
                    return self._tool(item.get("id"), item.get("tool") or item_type,
                                      item.get("arguments") or {"changes": item.get("changes", [])})
            if kind == "error":
                # A reconnect error can precede a successful turn.
                self._pending_error = str(event.get("message") or "Codex stream error")
            if kind == "turn.failed":
                self.errors.append(str(event.get("error") or "Codex turn failed"))
            if kind == "turn.completed":
                self.completed = True
                self._pending_error = ""
        elif self.backend == "opencode":
            part = _object(event.get("part"))
            if kind == "tool_use":
                name = part.get("tool", "tool")
                return self._tool(part.get("callID") or part.get("id"),
                                  "Bash" if name in {"bash", "shell"} else name,
                                  _object(part.get("state")).get("input"))
            if kind == "error":
                self.errors.append(str(event.get("error") or "OpenCode session failed"))
            if kind == "step_finish":
                identifier = part.get("id")
                if not identifier or identifier not in self._steps:
                    cost = part.get("cost")
                    if isinstance(cost, (int, float)):
                        self.cost = (self.cost or 0) + cost
                    if identifier:
                        self._steps.add(identifier)
                if part.get("reason") == "stop":
                    self.completed = True
        return []

    def failures(self):
        errors = list(self.errors)
        if self._pending_error:
            errors.append(self._pending_error)
        if not self.completed and not errors:
            errors.append(f"{self.backend} did not emit a successful completion event; see session log")
        return errors
