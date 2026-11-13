// Package agent implements the core agentic loop: send the conversation to
// Claude, execute whatever tools it asks for, feed the results back, repeat
// until Claude stops asking for tools (or we hit a turn cap). Everything
// else in this project (tools, data, tracing) exists to support this loop.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/richardknop/my-agent/internal/claude"
	"github.com/richardknop/my-agent/internal/data"
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

Once you have enough evidence, STOP calling tools and answer in plain text with exactly these
sections:

Root cause: <one or two sentences>
Evidence: <bullet-style list of the specific data points that support it — cite timestamps,
  services, metric values, deploy versions, or log messages>
Confidence: <low|medium|high, with a short reason>
Recommended action: <what you'd do next, e.g. rollback, alert threshold, follow-up query>

Be concise. Do not call a tool again once you have already gathered enough evidence to answer.`

// buildSystemPrompt appends the current time. Without it the model has no
// idea what "yesterday" or "an hour ago" means and will guess a date from
// its training data — a classic context-engineering gap. Real agents
// always need to be told "now".
func buildSystemPrompt(now time.Time) string {
	now = now.UTC()
	return systemPrompt + fmt.Sprintf(`

Current time: %s (%s). All timestamps in the tools are UTC. Resolve relative
times in the question ("yesterday afternoon", "an hour ago") against this current
time and pass explicit RFC3339 timestamps to the tools.`,
		now.Format(time.RFC3339), now.Weekday())
}

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

type toolDef = tools.Definition

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
		Model:     "claude-sonnet-4-5-20250929",
		MaxTokens: 1024,
		MaxTurns:  8,
	}
}

type Agent struct {
	client   LLM
	registry *tools.Registry
	store    *data.Store
	tracer   *trace.Tracer
	cfg      Config
	log      *stepLog
}

func New(client LLM, registry *tools.Registry, store *data.Store, tracer *trace.Tracer, cfg Config) *Agent {
	return &Agent{
		client: client, registry: registry, store: store, tracer: tracer, cfg: cfg,
		log: &stepLog{w: cfg.LogWriter, level: cfg.Verbosity},
	}
}

// Investigate runs the agent loop for one question and returns Claude's
// final text answer.
func (a *Agent) Investigate(ctx context.Context, question string) (string, error) {
	messages := []claude.Message{
		{Role: "user", Content: []claude.ContentBlock{{Type: "text", Text: question}}},
	}

	a.tracer.Event("run_start", map[string]any{"question": question, "model": a.cfg.Model, "now": a.cfg.Now})

	now := a.cfg.Now
	if now.IsZero() {
		now = time.Now()
	}
	system := buildSystemPrompt(now)
	defs := a.registry.Definitions()
	a.log.runStart(question, a.cfg.Model, now, system, defs)

	totalInputTokens, totalOutputTokens := 0, 0

	for turn := 1; turn <= a.cfg.MaxTurns; turn++ {
		req := claude.Request{
			Model:     a.cfg.Model,
			MaxTokens: a.cfg.MaxTokens,
			System:    system,
			Messages:  messages,
			Tools:     defs,
		}

		// Size of what we're about to send — useful for watching the context
		// window grow turn by turn.
		payload, err := json.Marshal(req)
		if err != nil {
			return "", a.notifyError(err, "marshal_error", nil, turn, totalInputTokens, totalOutputTokens)
		}
		a.log.request(turn, a.cfg.MaxTurns, len(messages), len(payload))

		start := time.Now()
		resp, err := a.client.CreateMessage(ctx, req)
		latency := time.Since(start)
		if err != nil {
			return "", a.notifyError(err, "llm_error", nil, turn, totalInputTokens, totalOutputTokens)
		}
		a.log.response(resp, latency, len(payload))

		totalInputTokens += resp.Usage.InputTokens
		totalOutputTokens += resp.Usage.OutputTokens
		a.tracer.Event("llm_response", map[string]any{
			"turn":          turn,
			"latency_ms":    latency.Milliseconds(),
			"stop_reason":   resp.StopReason,
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
			"request_bytes": len(payload),
		})

		messages = append(messages, claude.Message{Role: "assistant", Content: resp.Content})

		if resp.StopReason != "tool_use" {
			a.tracer.Event("run_end", map[string]any{
				"turns":               turn,
				"total_input_tokens":  totalInputTokens,
				"total_output_tokens": totalOutputTokens,
			})
			a.log.runEnd(turn, totalInputTokens, totalOutputTokens, nil)
			return textOf(resp.Content), nil
		}

		var resultBlocks []claude.ContentBlock
		for _, block := range resp.Content {
			if block.Type != "tool_use" {
				continue
			}

			toolStart := time.Now()
			result, callErr := a.registry.Call(block.Name, a.store, block.Input)
			toolLatency := time.Since(toolStart)

			isError := callErr != nil
			content := result
			if callErr != nil {
				content = callErr.Error()
			}

			a.tracer.Event("tool_call", map[string]any{
				"turn":       turn,
				"tool":       block.Name,
				"input":      string(block.Input),
				"latency_ms": toolLatency.Milliseconds(),
				"is_error":   isError,
				"result_len": len(content),
			})
			a.log.toolCall(block.Name, block.Input, result, callErr, toolLatency)

			resultBlocks = append(resultBlocks, claude.ContentBlock{
				Type:      "tool_result",
				ToolUseID: block.ID,
				Content:   content,
				IsError:   isError,
			})
		}

		if len(resultBlocks) > 0 {
			messages = append(messages, claude.Message{Role: "user", Content: resultBlocks})
			a.log.resultsAppended(len(resultBlocks))
		}
	}

	a.tracer.Event("run_end", map[string]any{
		"turns":               a.cfg.MaxTurns,
		"total_input_tokens":  totalInputTokens,
		"total_output_tokens": totalOutputTokens,
		"error":               "max turns exceeded",
	})
	err := fmt.Errorf("exceeded max turns (%d) without a final answer", a.cfg.MaxTurns)
	a.log.runEnd(a.cfg.MaxTurns, totalInputTokens, totalOutputTokens, err)
	return "", err
}

func (a *Agent) notifyError(err error, event string, extraData map[string]any, turn int, totalInputTokens, totalOutputTokens int) error {
	data := map[string]any{
		"turn":  turn,
		"error": err.Error(),
	}
	for k, v := range extraData {
		data[k] = v
	}
	a.tracer.Event(event, data)
	err = fmt.Errorf("turn %d: %w", turn, err)
	a.log.runEnd(turn, totalInputTokens, totalOutputTokens, err)
	return err
}

func textOf(blocks []claude.ContentBlock) string {
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}
