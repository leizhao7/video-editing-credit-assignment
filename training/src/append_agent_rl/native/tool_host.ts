/** Persistent tool-only host built inside the pinned OpenCode source tree.
 * No SessionPrompt.loop/process/LLM invocation: the Python append loop owns all
 * generation. SessionTools.resolve retains native schema/plugin/MCP behavior.
 */
import { Effect, Schema } from "effect"
import { asSchema } from "ai"
import { createInterface } from "node:readline"
import { AppRuntime } from "@/effect/app-runtime"
import { EffectBridge } from "@/effect/bridge"
import { InstanceStore } from "@/project/instance-store"
import { InstanceRef } from "@/effect/instance-ref"
import { InstanceState } from "@/effect/instance-state"
import { Agent } from "@/agent/agent"
import { Provider } from "@/provider/provider"
import { Permission } from "@/permission"
import { Plugin } from "@/plugin"
import { Config } from "@/config/config"
import { Session } from "@/session/session"
import { MessageV2 } from "@/session/message-v2"
import { MessageID, PartID } from "@/session/schema"
import { SessionTools } from "@/session/tools"
import { SystemPrompt } from "@/session/system"
import { Instruction } from "@/session/instruction"
import MAX_STEPS from "@/session/prompt/max-steps.txt"

const DENIED = new Set(["webfetch", "websearch", "task", "task_status", "question", "repo_clone",
  "repo_overview", "plan_enter", "plan_exit"])

const Call = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  arguments: Schema.Record(Schema.String, Schema.Unknown),
})
const Step = Schema.Struct({text: Schema.String, calls: Schema.Array(Call)})
type Step = Schema.Schema.Type<typeof Step>

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Exported for tests using a fresh isolated temporary InstanceRef. */
export const createHost = Effect.fn("AppendTools.create")(function* (task: string) {
  const run = yield* EffectBridge.make()
  const instance = yield* InstanceState.context
  const sessions = yield* Session.Service
  const plugin = yield* Plugin.Service
  const agents = yield* Agent.Service
  const provider = yield* Provider.Service
  const config = yield* Config.Service.use((svc) => svc.get())
  const agent = yield* agents.get("build")
  if (!agent) throw new Error("Missing pinned build agent")
  const ref = agent.model ?? (yield* provider.defaultModel())
  const model = yield* provider.getModel(ref.providerID, ref.modelID).pipe(Effect.orDie)
  const session = yield* sessions.create({title: "Append-only RL tool session"})
  const user: MessageV2.User = {id: MessageID.ascending(), sessionID: session.id, role: "user",
    agent: agent.name, model: ref, time: {created: Date.now()}}
  const userPart: MessageV2.TextPart = {id: PartID.ascending(), sessionID: session.id, messageID: user.id,
    type: "text", text: task}
  yield* sessions.updateMessage(user)
  yield* sessions.updatePart(userPart)
  const messages: MessageV2.WithParts[] = [{info: user, parts: [userPart]}]
  const parts = new Map<string, MessageV2.ToolPart>()
  const seen = new Set<string>()
  let busy = false
  let assistant: MessageV2.Assistant = newAssistant()

  function newAssistant(): MessageV2.Assistant {
    return {id: MessageID.ascending(), sessionID: session.id, role: "assistant", parentID: user.id,
      time: {created: Date.now()}, modelID: model.id, providerID: model.providerID, mode: "build",
      agent: agent.name, path: {cwd: instance.directory, root: instance.worktree}, cost: 0,
      tokens: {input: 0, output: 0, reasoning: 0, cache: {read: 0, write: 0}}}
  }

  const savePart = Effect.fn("AppendTools.savePart")(function* (part: MessageV2.ToolPart) {
    parts.set(part.callID, part)
    const message = messages.find((m) => m.info.id === part.messageID)
    if (!message) throw new Error("Tool part has no recorded assistant")
    const index = message.parts.findIndex((p) => p.id === part.id)
    if (index < 0) message.parts.push(part)
    if (index >= 0) message.parts[index] = part
    yield* sessions.updatePart(part)
    return part
  })

  const processor = {
    get message() { return assistant },
    updateToolCall: (id: string, update: (p: MessageV2.ToolPart) => MessageV2.ToolPart) =>
      parts.has(id) ? savePart(update(parts.get(id)!)) : Effect.succeed(undefined),
    completeToolCall: () => Effect.void,
  }
  // Defense in depth: these operations cannot accidentally start a native
  // model/subagent even if a future registry change exposes a task-like tool.
  const denied = () => Effect.die(new Error("Native agent/model loops are disabled in tool-only host"))
  const available = yield* SessionTools.resolve({agent, model, session, processor, bypassAgentCheck: false,
    messages, promptOps: {cancel: denied, prompt: denied, loop: denied, resolvePromptParts: denied}})
  const disabled = Permission.disabled(Object.keys(available), Permission.merge(agent.permission, session.permission ?? []))
  const tools = Object.fromEntries(Object.entries(available).filter(([name]) =>
    !DENIED.has(name) && !disabled.has(name) && config.tools?.[name] !== false))
  const catalogue = yield* Effect.promise(() => Promise.all(Object.entries(tools).map(async ([name, tool]) =>
    ({type: "function" as const, function: {name, description: tool.description ?? "",
      parameters: await Promise.resolve(asSchema(tool.inputSchema).jsonSchema)}}))))
  const sys = yield* Effect.gen(function* () {
    const service = yield* SystemPrompt.Service
    return {environment: yield* service.environment(model), skills: yield* service.skills(agent)}
  }).pipe(Effect.provide(SystemPrompt.defaultLayer))
  const instructions = yield* Instruction.Service.use((svc) => svc.system()).pipe(Effect.orDie)
  const system = [[...(agent.prompt ? [agent.prompt] : SystemPrompt.provider(model)),
    ...sys.environment, ...instructions, ...(sys.skills ? [sys.skills] : [])].filter(Boolean).join("\n")]
  yield* plugin.trigger("experimental.chat.system.transform", {sessionID: session.id, model}, {system})
  const profile = {format: "append-opencode-tools-v1", session_id: session.id,
    opencode_commit: "d7a6e1daaf2271bcd0b611fbb27d4956806e475a", system,
    task, tools: catalogue, final_turn_prompt: MAX_STEPS,
    native_agent_loop: false, tool_execution: "ordered", denied_tools: [...DENIED]}

  async function step(input: Step, signal = new AbortController().signal) {
    if (busy) throw new Error("One policy turn at a time per tool session")
    const action = Schema.decodeUnknownSync(Step)(input)
    if (!action.calls.length || action.calls.length > 64) throw new Error("Invalid tool call count")
    const current = new Set<string>()
    for (const call of action.calls) {
      if (!/^[A-Za-z0-9_-]{1,160}$/.test(call.id) || seen.has(call.id) || current.has(call.id))
        throw new Error("Duplicate/invalid tool call identity")
      current.add(call.id)
    }
    busy = true
    try {
      assistant = newAssistant()
      const text: MessageV2.TextPart = {id: PartID.ascending(), sessionID: session.id,
        messageID: assistant.id, type: "text", text: action.text}
      messages.push({info: assistant, parts: [text]})
      await run.promise(sessions.updateMessage(assistant))
      await run.promise(sessions.updatePart(text))
      const results = []
      for (const call of action.calls) {
        seen.add(call.id)
        const start = Date.now()
        const part: MessageV2.ToolPart = {id: PartID.ascending(), sessionID: session.id,
          messageID: assistant.id, type: "tool", callID: call.id, tool: call.name,
          state: {status: "running", input: call.arguments, time: {start}}}
        await run.promise(savePart(part))
        try {
          const tool = tools[call.name]
          if (!tool?.execute) throw new Error(`Tool ${call.name} is disabled or unavailable`)
          if (signal.aborted) throw new Error("Tool step aborted")
          const output: unknown = await tool.execute(call.arguments, {toolCallId: call.id,
            messages: [], abortSignal: signal})
          if (!record(output) || typeof output.output !== "string") throw new Error("Invalid native tool output")
          const attachments = output.attachments === undefined ? [] :
            Schema.decodeUnknownSync(Schema.Array(MessageV2.FilePart))(output.attachments)
          const normalized = {title: typeof output.title === "string" ? output.title : "",
            output: output.output, metadata: record(output.metadata) ? output.metadata : {},
            attachments: [...attachments]}
          await run.promise(savePart({...part, state: {status: "completed", input: call.arguments,
            ...normalized, time: {start, end: Date.now()}}}))
          results.push({id: call.id, name: call.name, is_error: false, ...normalized})
        } catch (error) {
          if (signal.aborted) throw error
          const message = error instanceof Error ? error.message : String(error)
          await run.promise(savePart({...part, state: {status: "error", input: call.arguments,
            error: message, time: {start, end: Date.now()}}}))
          results.push({id: call.id, name: call.name, is_error: true, title: "Tool error",
            output: message, metadata: {}, attachments: []})
        }
      }
      assistant.finish = "tool-calls"
      assistant.time.completed = Date.now()
      await run.promise(sessions.updateMessage(assistant))
      return {results}
    } finally {
      busy = false
    }
  }
  return {profile, step}
})

async function main() {
  if (process.argv[2] === "--version") {
    process.stdout.write("append-opencode-tools-v1 upstream=d7a6e1daaf2271bcd0b611fbb27d4956806e475a\n")
    return
  }
  if (process.getuid?.() === 0 || process.cwd() !== "/work" ||
      !await Bun.file("/opt/OCRL_SANDBOX_IMAGE").exists())
    throw new Error("Tool host is only for an unprivileged, network-blocked rollout sandbox")
  const lines = createInterface({input: process.stdin, crlfDelay: Infinity})
  const abort = new AbortController()
  process.on("SIGTERM", () => { abort.abort(); lines.close() })
  const {store, ctx} = await AppRuntime.runPromise(InstanceStore.Service.use((store) =>
    store.load({directory: "/work"}).pipe(Effect.map((ctx) => ({store, ctx})))))
  const host = await AppRuntime.runPromise(createHost(await Bun.file(process.argv[2]).text())
    .pipe(Effect.provideService(InstanceRef, ctx)))
  try {
    process.stdout.write(JSON.stringify({kind: "profile", profile: host.profile}) + "\n")
    for await (const line of lines) {
      if (line.length > 16 * 1024 * 1024) throw new Error("Oversized tool command")
      const frame: unknown = JSON.parse(line)
      if (!record(frame) || typeof frame.id !== "string" || frame.method !== "step")
        throw new Error("Unknown tool-host operation")
      const action = Schema.decodeUnknownSync(Step)(frame.action)
      const result = await host.step(action, abort.signal)
      process.stdout.write(JSON.stringify({kind: "tool_result", id: frame.id, result}) + "\n")
    }
  } finally {
    await AppRuntime.runPromise(store.dispose(ctx))
    await AppRuntime.dispose()
  }
}

if (import.meta.main) await main()
