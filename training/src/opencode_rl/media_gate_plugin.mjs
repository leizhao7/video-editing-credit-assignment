// Shared native/MCP output gate. Runs before OpenCode stores/converts tool images,
// so inference and training see the SAME normalized, budgeted observations.
import { createHash } from "node:crypto"

export const MediaGatePlugin = async () => ({
  "tool.execute.after": async (input, output) => {
    const slots = []
    for (const part of output.content ?? []) {
      if (part.type === "image") {
        slots.push({ url: `data:${part.mimeType};base64,${part.data}`, set: (url) => {
          part.mimeType = "image/jpeg"
          part.data = url.split(",", 2)[1]
        } })
      } else if (part.type === "resource" && part.resource?.blob) {
        if (!part.resource.mimeType?.startsWith("image/")) {
          throw new Error("Native audio/video/PDF observations are not wired; use the media inspection tools")
        }
        slots.push({ url: `data:${part.resource.mimeType};base64,${part.resource.blob}`, set: (url) => {
          part.resource.mimeType = "image/jpeg"
          part.resource.blob = url.split(",", 2)[1]
        } })
      }
    }
    for (const part of output.attachments ?? []) {
      if (!part.mime?.startsWith("image/")) {
        throw new Error("Native audio/video/PDF observations are not wired; use the media inspection tools")
      }
      slots.push({ url: part.url, set: (url) => {
        part.mime = "image/jpeg"
        part.url = url
      } })
    }
    if (!slots.length) return
    const identity = createHash("sha256").update(JSON.stringify([
      input.sessionID, input.callID, input.tool,
    ])).digest("hex")
    const response = await fetch("http://127.0.0.1:18080/ocrl/images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call_id: identity, images: slots.map((slot) => slot.url),
        source_units: output._meta?.ocrl_visual_source_units ?? 0 }),
      signal: AbortSignal.timeout(30000),
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.error ?? "Shared image admission failed")
    if (!Array.isArray(result.images) || result.images.length !== slots.length) {
      throw new Error("Shared image gate changed image cardinality")
    }
    slots.forEach((slot, i) => slot.set(result.images[i]))
    const budget = { charged_units: result.charged_units, spent_units: result.spent_units,
      limit_units: result.limit_units }
    if (Array.isArray(output.content)) {
      output.content.push({ type: "text", text: `Visual budget: ${JSON.stringify(budget)}` })
    } else {
      output.output += `\nVisual budget: ${JSON.stringify(budget)}`
    }
  },
})
