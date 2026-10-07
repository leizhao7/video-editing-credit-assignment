// Install at packages/opencode/script/build_append_host.ts. No release/upload,
// provider/model fetch, dependency installation or edits to the upstream loop.
import path from "node:path"
import { readdir, readFile } from "node:fs/promises"

const root = path.resolve(import.meta.dirname, "..")
process.chdir(root)
const target = process.argv[2] ?? "bun-linux-x64"
if (!["bun-linux-x64", "bun-darwin-arm64"].includes(target)) throw new Error("Unreviewed build target")
const entries = (await readdir("migration", {withFileTypes: true}))
  .filter((entry) => entry.isDirectory() && /^\d{14}/.test(entry.name)).map((entry) => entry.name).sort()
const migrations = await Promise.all(entries.map(async (name) => {
  const match = /^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/.exec(name)!
  return {name, sql: await readFile(path.join("migration", name, "migration.sql"), "utf8"),
    timestamp: Date.UTC(Number(match[1]), Number(match[2])-1, Number(match[3]),
      Number(match[4]), Number(match[5]), Number(match[6]))}
}))
const result = await Bun.build({
  entrypoints: ["./src/append_agent_rl/tool_host.ts"],
  target: "bun", conditions: ["browser"], tsconfig: "./tsconfig.json", format: "esm",
  external: ["node-gyp"], minify: true, metafile: true,
  compile: {target: target as "bun-linux-x64" | "bun-darwin-arm64",
    outfile: `./dist/append-tools/${target}/opencode-tools`,
    autoloadBunfig: false, autoloadDotenv: false, autoloadTsconfig: false, autoloadPackageJson: false},
  define: {OPENCODE_VERSION: JSON.stringify("1.15.5-append-tools-v1"),
    OPENCODE_CHANNEL: JSON.stringify("local"), OPENCODE_MIGRATIONS: JSON.stringify(migrations),
    // Only the explicit pinned local provider is enabled in rollout config.
    // No online model registry is loaded during a reproducible tool-only build.
    OPENCODE_MODELS_DEV: "{}", OPENCODE_LIBC: JSON.stringify(target === "bun-linux-x64" ? "glibc" : "")},
})
if (!result.success) throw new AggregateError(result.logs, "Tool-only binary build failed")
if (!result.metafile) throw new Error("Missing dependency bundle provenance")
const inputs = Object.keys(result.metafile.inputs)
if (inputs.some((name) => name.includes("ghostty"))) throw new Error("Unpinned web-app dependency entered tool host")
await Bun.write(`./dist/append-tools/${target}/bundle-metafile.json`, JSON.stringify(result.metafile, null, 2))
for (const output of result.outputs) {
  const body = await Bun.file(output.path).arrayBuffer()
  const sha256 = new Bun.CryptoHasher("sha256").update(body).digest("hex")
  console.log(JSON.stringify({path: output.path, bytes: body.byteLength, sha256, target,
    source: "d7a6e1daaf2271bcd0b611fbb27d4956806e475a", native_agent_loop: false,
    bundle_inputs: inputs.length, ghostty_dependency: false}))
}
