// Package agent implements the core agentic loop: send the conversation to
// Claude, execute whatever tools it asks for, feed the results back, repeat
// until Claude submits its report through the submit_report tool (or we hit
// a turn cap). Everything else in this project (tools, data, tracing) exists
// to support this loop.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/richardknop/my-agent/internal/claude"
	"github.com/richardknop/my-agent/internal/tools"
	"github.com/richardknop/my-agent/internal/trace"
)

const systemPrompt = `You are an incident-investigation agent for an e-commerce platform.

You have read-only tools to inspect the system: list_services, query_metrics, query_logs, and
get_deploys. Use them to gather evidence before concluding anything — do not guess.

A good investigation typically:
1. Calls list_services to understand the dependency graph.
2. Uses query_metrics to confirm which service(s) actually show anomalous error_rate or
   p99_latency_ms, and pin down the time window when it started.
3. Uses get_deploys around that time window — deploys are the most common root cause.
4. Uses query_logs (narrowed by service, level and time range) to find the specific error
   messages that explain the mechanism, not just that something failed.
5. Follows the dependency graph: if a downstream service (e.g. payments) is impacted, check
   whether that explains anomalies in the services that call it (e.g. checkout, frontend),
   rather than treating each anomaly as independent.

Once you have enough evidence, STOP investigating and answer by calling submit_report. That
call is the only way to answer: plain-text answers are not recorded. In the report, cite the
specific data points that support your conclusion (timestamps, services, metric values, deploy
versions, log messages).

Be concise. Do not call a tool again once you have already gathered enough evidence to answer.`

// buildSystemPrompt appends what the model can't know on its own.
//
// The current time: without it the model has no idea what "yesterday" or
// "an hour ago" means and will guess a date from its training data — a
// classic context-engineering gap. Real agents always need to be told "now".
//
// The turn budget: without it the model has no reason to converge, and a
// thorough small model happily keeps checking one more service until the
// loop's turn cap cuts it off with nothing to show.
func buildSystemPrompt(now time.Time, maxTurns int) string {
	now = now.UTC()
	return systemPrompt + fmt.Sprintf(`

Current time: %s (%s). All timestamps in the tools are UTC. Resolve relative
times in the question ("yesterday afternoon", "an hour ago") against this current
time and pass explicit RFC3339 timestamps to the tools.

Turn budget: you have at most %d turns (a turn is one response from you). On the last
turn only submit_report is allowed, so plan to finish the investigation within %d
rounds of tool calls. Prefer answering with the evidence you have over
one more confirmation.`,
		now.Format(time.RFC3339), now.Weekday(), maxTurns, max(maxTurns-1, 0))
}

// finalTurnNote is appended to the last tool results before the final turn.
// It's the part that works with every server: Anthropic also honours a
// tool_choice that forces submit_report, but Ollama accepts and ignores it.
const finalTurnNote = "[agent harness] Turn budget exhausted: this is your final turn. Do not " +
	"call any investigation tools. Call submit_report now, using only the evidence gathered so " +
	"far; if it is incomplete, say what is missing and lower your confidence accordingly."

// submitReminder answers a plain-text reply. Models trained to "just answer"
// often do that even when told to use a tool; the harness enforces the
// output contract by asking again rather than accepting the text.
const submitReminder = "[agent harness] Your reply was not recorded as an answer: answers are only " +
	"accepted through the submit_report tool. If you have finished investigating, call " +
	"submit_report now; otherwise continue the investigation."

type Config struct {
	Model     string
	MaxTokens int
	MaxTurns  int
	// Now is the "current time" the agent is told about. Zero means
	// time.Now(); set it to replay a fixed dataset as if it were live.
	Now time.Time
	// Verbosity controls the human-readable step log written to LogWriter:
	// 0 silent, 1 each step, 2 each step plus full payloads.
	Verbosity int
	LogWriter io.Writer
}

// LLM is the one thing the loop needs from a model: send a Messages API
// request, get a response. It's defined here, on the consumer side, so the
// loop doesn't depend on the concrete HTTP client — main passes a
// *claude.Client, tests pass a scripted fake that replays canned responses
// and records every request the loop sent.
type LLM interface {
	CreateMessage(ctx context.Context, req claude.Request) (*claude.Response, error)
}

func DefaultConfig() Config {
	return Config{
		Model: "claude-sonnet-5-5",
		// Caps the length of each model response. The final report is
		// the longest one; a cap that cuts it off is now an error rather
		// than a truncated "answer", so leave generous headroom.
		MaxTokens: 4096,
		MaxTurns:  8,
	}
}

// Agent runs the loop. Note what it does *not* know about: the dataset.
// Tools are just "list the definitions, call one by name", so the loop
// works the same whether a tool reads in-memory JSON, queries ClickHouse,
// or (stage 3) is served by an MCP server.
type Agent struct {
	client   LLM
	registry *tools.Registry
	tracer   *trace.Tracer
	cfg      Config
	log      *stepLog
}

func New(client LLM, registry *tools.Registry, tracer *trace.Tracer, cfg Config) *Agent {
	return &Agent{
		client: client, registry: registry, tracer: tracer, cfg: cfg,
		log: &stepLog{w: cfg.LogWriter, level: cfg.Verbosity},
	}
}

// runStats accumulates what the end-of-run record reports.
type runStats struct {
	turns           int
	inputTokens     int
	cacheReadTokens int
	outputTokens    int
	// budgetExhausted: the report came on the final turn, after the model
	// was told to stop investigating. Worth tracking (and, in evals,
	// alerting on): it means the investigation was cut short.
	budgetExhausted bool
}

// Investigate runs the agent loop for one question and returns the report
// the model submitted. Every run, however it ends (report, model API error,
// turn cap), goes through finish exactly once, so the trace always ends with
// a run_end event and the step log always prints a summary.
func (a *Agent) Investigate(ctx context.Context, question string) (*Report, error) {
	var stats runStats
	report, err := a.run(ctx, question, &stats)
	a.finish(stats, report, err)
	return report, err
}

func (a *Agent) run(ctx context.Context, question string, stats *runStats) (*Report, error) {
	messages := []claude.Message{
		{Role: "user", Content: []claude.ContentBlock{{Type: "text", Text: question}}},
	}

	a.tracer.Event("run_start", map[string]any{"question": question, "model": a.cfg.Model, "now": a.cfg.Now})

	now := a.cfg.Now
	if now.IsZero() {
		now = time.Now()
	}
	system := buildSystemPrompt(now, a.cfg.MaxTurns)
	apiTools := append(toAPITools(a.registry.Definitions()), submitReportTool)
	a.log.runStart(question, a.cfg.Model, now, system, apiTools)

	for turn := 1; turn <= a.cfg.MaxTurns; turn++ {
		stats.turns = turn
		final := turn == a.cfg.MaxTurns
		req := claude.Request{
			Model:     a.cfg.Model,
			MaxTokens: a.cfg.MaxTokens,
			System:    system,
			Messages:  messages,
			Tools:     apiTools,
		}
		if final {
			// All tools must still be sent: the conversation contains
			// tool_use/tool_result blocks, which the API rejects without
			// their definitions. This tool_choice forces submit_report.
			req.ToolChoice = &claude.ToolChoice{Type: "tool", Name: submitReportToolName}
		}

		// Size of what we're about to send — useful for watching the context
		// window grow turn by turn.
		payload, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("turn %d: marshal request: %w", turn, err)
		}
		a.log.request(turn, a.cfg.MaxTurns, len(messages), len(payload))

		start := time.Now()
		resp, err := a.client.CreateMessage(ctx, req)
		latency := time.Since(start)
		if err != nil {
			a.tracer.Event("llm_error", map[string]any{
				"turn":       turn,
				"latency_ms": latency.Milliseconds(),
				"error":      err.Error(),
			})
			return nil, fmt.Errorf("turn %d: %w", turn, err)
		}
		a.log.response(resp, latency, len(payload))

		stats.inputTokens += resp.Usage.InputTokens
		stats.cacheReadTokens += resp.Usage.CacheReadInputTokens
		stats.outputTokens += resp.Usage.OutputTokens
		a.tracer.Event("llm_response", map[string]any{
			"turn":                    turn,
			"latency_ms":              latency.Milliseconds(),
			"stop_reason":             resp.StopReason,
			"input_tokens":            resp.Usage.InputTokens,
			"cache_read_input_tokens": resp.Usage.CacheReadInputTokens,
			"output_tokens":           resp.Usage.OutputTokens,
			"request_bytes":           len(payload),
		})

		if resp.StopReason == "max_tokens" {
			// The response was cut off mid-sentence (or mid tool call).
			// Returning it as the answer would pass half a report off as
			// a finished one, so fail loudly instead.
			return nil, fmt.Errorf("turn %d: response cut off at max_tokens (%d): raise -max-tokens",
				turn, a.cfg.MaxTokens)
		}

		messages = append(messages, claude.Message{Role: "assistant", Content: resp.Content})

		var toolUses []claude.ContentBlock
		for _, block := range resp.Content {
			if block.Type == "tool_use" {
				toolUses = append(toolUses, block)
			}
		}

		// A valid report ends the run, even if the model asked for other
		// tools in the same response: their results would go unread.
		// Invalid reports are kept to be answered with a tool error below.
		reportErrs := map[string]error{}
		var firstReportErr error
		for _, block := range toolUses {
			if block.Name != submitReportToolName {
				continue
			}
			report, err := a.checkReport(turn, block)
			if err == nil {
				stats.budgetExhausted = final && a.cfg.MaxTurns > 1
				return report, nil
			}
			reportErrs[block.ID] = err
			if firstReportErr == nil {
				firstReportErr = err
			}
		}

		if final {
			// The model ignored the instruction to submit. Running tools or
			// asking again would only cost turns the budget doesn't have.
			if firstReportErr != nil {
				return nil, fmt.Errorf("turn %d: the model submitted an invalid report on the final turn: %w", turn, firstReportErr)
			}
			if len(toolUses) > 0 {
				return nil, fmt.Errorf("exceeded max turns (%d) without a report: "+
					"the model kept calling tools on the final turn", a.cfg.MaxTurns)
			}
			return nil, fmt.Errorf("exceeded max turns (%d) without a report: "+
				"the model answered in plain text instead of calling %s", a.cfg.MaxTurns, submitReportToolName)
		}

		var next []claude.ContentBlock
		for _, block := range toolUses {
			if err, ok := reportErrs[block.ID]; ok {
				next = append(next, claude.ContentBlock{
					Type: "tool_result", ToolUseID: block.ID, Content: err.Error(), IsError: true,
				})
				continue
			}
			next = append(next, a.callTool(ctx, turn, block))
		}
		if len(toolUses) == 0 {
			next = append(next, claude.ContentBlock{Type: "text", Text: submitReminder})
			a.log.submitReminder()
		}
		if turn+1 == a.cfg.MaxTurns {
			// Text blocks may follow tool_result blocks in the same user
			// message, so the note rides along with the results.
			next = append(next, claude.ContentBlock{Type: "text", Text: finalTurnNote})
			a.log.finalTurnNote()
		}
		messages = append(messages, claude.Message{Role: "user", Content: next})
		a.log.resultsAppended(len(toolUses))
	}

	// Unreachable while the final turn always returns, but keeps the
	// compiler (and a future edit to the loop) honest.
	return nil, fmt.Errorf("exceeded max turns (%d) without a report", a.cfg.MaxTurns)
}

// checkReport validates a submit_report call. It's traced like any other
// tool call, so the trace shows rejected reports and what was wrong.
func (a *Agent) checkReport(turn int, block claude.ContentBlock) (*Report, error) {
	report, err := parseReport(block.Input)
	fields := map[string]any{
		"turn":       turn,
		"tool":       block.Name,
		"input":      string(block.Input),
		"latency_ms": 0,
		"is_error":   err != nil,
	}
	if err != nil {
		fields["error"] = err.Error()
	}
	a.tracer.Event("tool_call", fields)
	a.log.reportChecked(err)
	return report, err
}

// callTool runs one tool_use block and turns the outcome into the matching
// tool_result block. A failing tool is not a failed run: the error text goes
// back to the model with is_error set, so it can correct itself.
func (a *Agent) callTool(ctx context.Context, turn int, block claude.ContentBlock) claude.ContentBlock {
	start := time.Now()
	result, err := a.registry.Call(ctx, block.Name, block.Input)
	latency := time.Since(start)

	content := result
	if err != nil {
		content = err.Error()
	}

	a.tracer.Event("tool_call", map[string]any{
		"turn":       turn,
		"tool":       block.Name,
		"input":      string(block.Input),
		"latency_ms": latency.Milliseconds(),
		"is_error":   err != nil,
		"result_len": len(content),
	})
	a.log.toolCall(block.Name, block.Input, result, err, latency)

	return claude.ContentBlock{
		Type:      "tool_result",
		ToolUseID: block.ID,
		Content:   content,
		IsError:   err != nil,
	}
}

// finish writes the end-of-run record to the trace and the step log.
func (a *Agent) finish(stats runStats, report *Report, err error) {
	fields := map[string]any{
		"turns":                         stats.turns,
		"total_input_tokens":            stats.inputTokens,
		"total_cache_read_input_tokens": stats.cacheReadTokens,
		"total_output_tokens":           stats.outputTokens,
		"budget_exhausted":              stats.budgetExhausted,
	}
	if report != nil {
		fields["report"] = report
	}
	if err != nil {
		fields["error"] = err.Error()
	}
	a.tracer.Event("run_end", fields)
	a.log.runEnd(stats, err)
}

// toAPITools converts our tool definitions into the Messages API's tool
// type. They have the same fields today, but they're separate types on
// purpose: tools describes what a tool is, claude describes what goes on the
// wire to one particular model API. An MCP server (stage 3) would describe
// the same tools in its own protocol's shape.
func toAPITools(defs []tools.Definition) []claude.Tool {
	out := make([]claude.Tool, len(defs))
	for i, d := range defs {
		out[i] = claude.Tool{Name: d.Name, Description: d.Description, InputSchema: d.InputSchema}
	}
	return out
}
