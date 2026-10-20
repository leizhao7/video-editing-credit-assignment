# Set up an annotation agent

Choose **Claude Code, Codex CLI, or OpenCode 1.x** to create task annotations. Use whichever CLI and model you already have access to. This guide covers the annotation backend; [RL training and its rollout harness](training.md) have separate setup requirements.

The setup helper needs only Python's standard library. Video preprocessing dependencies, FFmpeg, and cloud stages are described in the [task-generation guide](data_construction.md).

## 1. Install one CLI and sign in

Install just the backend you want to use. Follow its official installation guide, then authenticate through that CLI:

| Backend | Installation | Sign in | Check configuration / select a model |
| --- | --- | --- | --- |
| Claude Code | [Official setup](https://code.claude.com/docs/en/setup) | `claude auth login` | `claude auth status`; use `/model` in an interactive Claude session |
| Codex CLI | [Official setup](https://developers.openai.com/codex/cli) | `codex login` | `codex login status`; use `/model` in an interactive Codex session |
| OpenCode 1.x | [Official CLI documentation](https://opencode.ai/docs/cli/) | `opencode auth login` | `opencode auth list`; list model IDs with `opencode models` |

For the OpenCode version tested here, install the explicit 1.x package version (requires Node.js/npm):

```sh
npm install -g opencode-ai@1.19.3
opencode auth login
opencode models
```

Keep an existing working login. Credentials stay in the CLI's native authentication system; the repository setup helper never requests or copies API keys or login tokens.

Use a model that supports **image input and tool use**. OpenCode model IDs use `provider/model`. Version, login-status, and model-list checks do not establish successful inference; the small live check below verifies that configuration. [Previously tested versions and models](data_construction.md#real-cli-checks) are recorded for reference. OpenCode 2.x is not yet supported by this adapter.

## 2. Create your backend configuration

From the repository root:

```sh
python3 data_construction/setup_agent.py
```

The wizard asks for the backend, model ID, private data directory, and optional executable path and effort. It checks the executable version and saves a profile such as:

```text
~/.config/video-editing-credit-assignment/codex.env
```

Copy the `source` command printed by the wizard into your current terminal. For example:

```sh
source ~/.config/video-editing-credit-assignment/codex.env
```

Profiles are shell configuration for Bash/Zsh. They live outside the repository and contain only backend settings and paths. The helper defaults private media/output storage to `~/video-editing-data`, also outside the repository. It does not process media or call a model during ordinary setup. An existing profile is preserved unless you explicitly pass `--force`.

For a noninteractive setup, supply the choices directly:

```sh
python3 data_construction/setup_agent.py \
  --backend codex \
  --model '<available-model-id>' \
  --data-root "$HOME/video-editing-data"
```

Optional flags: `--cli '/absolute/path/to/executable'`, `--effort '<native-effort>'`, and `--output '/outside/repository/custom.env'`. Use `--help` for details. Missing CLI installation is reported with next steps; saving a profile alone does not establish that the backend is ready.

## 3. Verify before processing media

Run the check command printed by the wizard. It includes your selected backend, model, executable, and effort, and calls `check_annotation_cli.py` on temporary generated inputs. You can also add `--check` when creating a profile:

```sh
python3 data_construction/setup_agent.py \
  --backend opencode \
  --model '<provider/model>' \
  --check
```

**The live check calls a real model and may incur usage.** It asks the agent to read a small JSON file, view a generated two-color image, and write a result. It checks the output, unchanged inputs, exit code, and completion events. Logs stay in the external temporary directory printed by the checker. Confirm actual image-tool use in the trace before relying on a model for visual annotation; a correct color answer alone is insufficient evidence.

A passing check establishes this small CLI task. Media preprocessing, FFmpeg permissions, optional MCP tools, and annotation quality still need checking for your own pipeline.

Some Codex image calls are absent from its JSON event stream. In that case, inspect the corresponding native CLI session trace before concluding that image viewing succeeded or failed; the generated color answer alone is not enough.

## 4. Run annotation or switch backends

After preprocessing has produced the required inputs, activate a profile and use the existing entry points:

```sh
source ~/.config/video-editing-credit-assignment/codex.env
python3 data_construction/run_script_queue.py --film '<your-film-slug>' --prep-only
python3 data_construction/run_script_queue.py --film '<your-film-slug>'
```

The first command prepares annotation inputs; the second starts the model. Continue with the [documented pipeline sequence](data_construction.md#pipeline-and-explicit-entry-points).

Create another backend's profile with the wizard, then switch in the same terminal:

```sh
source ~/.config/video-editing-credit-assignment/claude.env
# Or, after creating an OpenCode profile:
source ~/.config/video-editing-credit-assignment/opencode.env
```

Each profile explicitly sets or clears the executable override, effort, and Claude-specific MCP setting, so switching does not inherit those values from the previous backend. Optional S4 MCP integration can be configured afterward using the [backend-specific rules](data_construction.md#annotation-cli-configuration). Other native CLI settings, including authentication and permission rules, remain managed by that CLI.

Changing backends does not automatically regenerate completed annotations. For comparisons, choose separate private data directories; the script-annotation queue also provides `--redo`.

## Troubleshooting

| What you see | Next step |
| --- | --- |
| Executable not found | Install the selected CLI, reopen the shell if needed, or configure its absolute path with `--cli`. |
| OpenCode 2.x detected | Use a supported OpenCode 1.x installation and select its executable with `--cli`. |
| Login or model-access error | Check native login status and select a model available to that account. The adapter does not silently switch models. |
| Model cannot view the image | Choose a vision-capable model and enable the CLI's local image-reading tool. |
| Permission denied or no result file | Inspect the local logs and native permission configuration. The task directory needs write access, and reference-media links need read access. |
| Profile already exists | Source the existing profile, choose a new `--output` path, or deliberately replace it with `--force`. |
| A different backend still runs | Source the intended profile in the same shell used to launch the pipeline. Profiles are not loaded automatically. |

For all environment variables, see [annotation configuration](data_construction.md#annotation-cli-configuration). For model/CLI compatibility already exercised by this repository, see [real CLI checks](data_construction.md#real-cli-checks).
