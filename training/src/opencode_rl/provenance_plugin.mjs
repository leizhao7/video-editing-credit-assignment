// Pinned with the runtime configuration in both rollout and evaluation.
// Only headers are added; OpenCode retains its own prompts/tools/compaction.
import { randomUUID } from "node:crypto"

export const ProvenancePlugin = async () => {
  const turns = new Map()
  const contract = process.env.OCRL_CONTRACT_SHA256
  const attempt = process.env.OCRL_ATTEMPT_ID
  if (!/^[a-f0-9]{64}$/.test(contract ?? "") || !/^[A-Za-z0-9_-]{1,160}$/.test(attempt ?? "")) {
    throw new Error("Missing OpenCode RL provenance contract")
  }
  return {
    "chat.headers": async (input, output) => {
      const primary = input.agent === "build"
      const turn = primary ? (turns.get(input.sessionID) ?? 0) : 0
      if (primary) turns.set(input.sessionID, turn + 1)
      Object.assign(output.headers, {
        "x-ocrl-contract": contract,
        "x-ocrl-attempt": attempt,
        "x-ocrl-session": input.sessionID,
        "x-ocrl-call": randomUUID(),
        "x-ocrl-agent": input.agent,
        "x-ocrl-primary": primary ? "1" : "0",
        "x-ocrl-turn": String(turn),
      })
    },
  }
}
