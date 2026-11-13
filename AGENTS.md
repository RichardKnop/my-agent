# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, ...) working in this repo.
The human-facing overview, design rationale and the staged roadmap live in
[README.md](README.md); read its "Design decisions" and "Roadmap" sections before
starting a roadmap stage.

## What this project is

A **learning project**: an incident-investigation agent (answers "why is production
broken?" over mock logs, metrics and deploys) built from first principles so every
layer of a tool-using LLM agent is visible: the agentic loop, tool design, context
engineering, tracing, cost. The point is understanding, so clarity beats features.

## Ground rules

- **Standard library only in production code.** No Anthropic SDK, no agent framework,
  no MCP library. When a stage needs one (MCP's JSON-RPC, a ClickHouse client),
  hand-roll the minimal version. The one allowed dependency is
  `github.com/stretchr/testify`, and only in tests. Ask before adding any other.
- **One roadmap stage at a time.** Don't build ahead of the current stage or add
  speculative abstractions "for later".
- **Comments explain why and which agent concept the code shows**, not what the code
  does. Follow the existing style: each package doc comment ties the package to a
  concept and, where relevant, a roadmap stage.
- **Prompts and tool descriptions are behavior.** The system prompt (`internal/agent`)
  and tool `Description` strings (`internal/tools`) are read by the model. Treat
  changes to them as behavior changes and mention them in your summary.
- **Don't leak the answer.** Never put the incident's root cause (the payments deploy,
  the pool timeout) into the system prompt, tool descriptions or skills. The agent has
  to find it from the data, or the future evals stage measures nothing.
- **Keep README in sync** when flags, tool behavior, trace fields or step-log output
  change.

## Layout

```
cmd/agent/          CLI entrypoint; wires client, tools, store, tracer, runs one question
internal/agent/     the agentic loop + system prompt (agent.go), stderr narration (steplog.go)
internal/claude/    raw-HTTP Anthropic Messages API client (also works with Ollama)
internal/tools/     tool JSON-schema definitions + handlers, Registry for name lookup
internal/data/      loads data/*.json into an in-memory Store
internal/trace/     one JSONL event per LLM/tool call, written to traces/
scripts/gen_data.py generates the mock dataset
skills/             stage 2 placeholder
```

## Commands

```
python3 scripts/gen_data.py                    # regenerate data/*.json (incident = yesterday, UTC)
python3 scripts/gen_data.py --date 2026-08-23  # pinned, reproducible dataset
go build ./... && go vet ./... && go test ./...
go run ./cmd/agent -v 2 "why did checkout error rate spike yesterday afternoon?"
```

- Tool tests run against the real `data/*.json` and work out the incident day from
  the `payments v2.14.0` deploy. Never hardcode dates in those tests.
- `data/*.json` is committed. Regenerating it changes every timestamp, so don't commit
  regenerated data unless that's the intent.
- Running the agent calls a real model: against Anthropic it needs `ANTHROPIC_API_KEY`
  and costs money, so don't run it unless asked. A local Ollama model works via
  `-base-url http://localhost:11434 -model qwen3-coder` (see README).
- `-v 2` prints everything that enters the context window (system prompt, tool
  definitions, tool results). Use it when debugging agent behavior.

## Testing conventions

- Use testify: `require` for preconditions whose failure makes the rest of the test
  meaningless (errors, lengths before indexing), `assert` for everything else.
- Define interfaces on the consumer side (e.g. `agent.LLM`), and give them only the
  methods the consumer calls.
- Fake external boundaries (the model API) with small hand-written fakes that replay
  scripted responses and record requests. See `fakeLLM` in
  `internal/agent/agent_test.go`. Don't fake our own code: loop tests use the real
  tool registry against a tiny in-memory `data.Store`.
- Tests never call a real model or the network.

## The mock incident (for writing tests, not prompts)

A `payments` deploy (`v2.14.0`, ~14:31 UTC on the incident day) drops the DB pool
timeout from 5000ms to 500ms. Payments times out acquiring connections, `checkout`
(a synchronous caller) errors and retries, and `frontend` returns 503s. A rollback
(`v2.14.1`) fixes it. `inventory` / `inventory-db` stay healthy as a control group.
