package agent

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/richardknop/my-agent/internal/claude"
)

// stepLog narrates the agent loop in human-readable form as it runs, so you
// can watch each step: what is sent to the model, what it decides, which
// tools run, and what goes back into the conversation. It complements the
// JSONL trace (internal/trace), which is the machine-readable record.
//
// Verbosity levels:
//
//	0  silent
//	1  one block per step: requests, responses, tool calls, sizes, tokens
//	2  level 1 plus full payloads: system prompt, tool definitions, tool
//	   results — everything that actually enters the context window
type stepLog struct {
	w     io.Writer
	level int
}

func (l *stepLog) on(level int) bool { return l.w != nil && l.level >= level }

func (l *stepLog) printf(format string, args ...any) { fmt.Fprintf(l.w, format, args...) }

func (l *stepLog) runStart(question, model string, now time.Time, system string, defs []claude.Tool) {
	if !l.on(1) {
		return
	}
	names := make([]string, len(defs))
	for i, d := range defs {
		names[i] = d.Name
	}
	l.printf("━━━━ run start ━━━━\n")
	l.printf("question : %s\n", question)
	l.printf("model    : %s\n", model)
	l.printf("now      : %s (injected into the system prompt)\n", now.UTC().Format(time.RFC3339))
	l.printf("tools    : %s\n", strings.Join(names, ", "))
	if l.on(2) {
		l.printf("\nsystem prompt (%s, sent on every turn):\n%s\n", size(len(system)), indent(system))
		for _, d := range defs {
			l.printf("\ntool definition %q (sent on every turn):\n%s\n", d.Name, indent(d.Description))
		}
	}
}

func (l *stepLog) request(turn, maxTurns, nMessages, payloadBytes int) {
	if !l.on(1) {
		return
	}
	l.printf("\n──── turn %d/%d ────\n", turn, maxTurns)
	l.printf("→ LLM request : %d message(s) in conversation, %s payload\n", nMessages, size(payloadBytes))
	l.printf("                (the whole conversation is re-sent every turn; the model is stateless)\n")
}

func (l *stepLog) response(resp *claude.Response, latency time.Duration, payloadBytes int) {
	if !l.on(1) {
		return
	}
	u := resp.Usage
	cached := ""
	if u.CacheReadInputTokens > 0 {
		cached = fmt.Sprintf(" (+%d read from cache)", u.CacheReadInputTokens)
	}
	l.printf("← LLM response: %s, stop_reason=%s, tokens in=%d%s out=%d\n",
		latency.Round(10*time.Millisecond), resp.StopReason, u.InputTokens, cached, u.OutputTokens)

	if bpt, ok := truncationSuspected(payloadBytes, u); ok {
		l.printf("  ! %s sent but only %d prompt tokens counted (~%d bytes/token): the server may have\n"+
			"    truncated the context. With Ollama, raise OLLAMA_CONTEXT_LENGTH.\n",
			size(payloadBytes), promptTokens(u), bpt)
	}

	for _, b := range resp.Content {
		switch b.Type {
		case "text":
			if t := strings.TrimSpace(b.Text); t != "" && resp.StopReason == "tool_use" {
				l.printf("  model says  : %s\n", oneLine(t, 400))
			}
		case "tool_use":
			l.printf("  model wants : %s(%s)  [id %s]\n", b.Name, compact(b.Input), b.ID)
		}
	}

	switch resp.StopReason {
	case "tool_use":
		// the caller logs the tool executions next
	case "end_turn":
		l.printf("  → no tool calls: the model is done, its text is the final answer\n")
	case "max_tokens":
		l.printf("  ! stop_reason=max_tokens: the answer was cut off by the MaxTokens limit\n")
	default:
		l.printf("  → stop_reason %q treated as final\n", resp.StopReason)
	}
}

func (l *stepLog) toolCall(name string, input []byte, result string, err error, latency time.Duration) {
	if !l.on(1) {
		return
	}
	if err != nil {
		l.printf("⚙ tool %s(%s) → ERROR after %s: %v\n", name, compact(input), latency, err)
		l.printf("  (the error goes back to the model as an is_error tool_result so it can correct itself)\n")
		return
	}
	l.printf("⚙ tool %s(%s) → %s result in %s\n", name, compact(input), size(len(result)), latency.Round(time.Microsecond))
	if l.on(2) {
		l.printf("%s\n", indent(truncate(result, 3000)))
	}
}

func (l *stepLog) resultsAppended(n int) {
	if !l.on(1) {
		return
	}
	l.printf("→ appended %d tool_result block(s) as a new user message; looping back to the LLM\n", n)
}

func (l *stepLog) finalTurnNote() {
	if !l.on(1) {
		return
	}
	l.printf("→ next turn is the last: added a note telling the model to answer without tools\n")
}

func (l *stepLog) runEnd(stats runStats, err error) {
	if !l.on(1) {
		return
	}
	l.printf("\n━━━━ run end ━━━━\n")
	switch {
	case err != nil:
		l.printf("result   : FAILED: %v\n", err)
	case stats.budgetExhausted:
		l.printf("result   : final answer after %d turn(s), forced by the turn budget (investigation was cut short)\n", stats.turns)
	default:
		l.printf("result   : final answer after %d turn(s)\n", stats.turns)
	}
	l.printf("tokens   : in=%d cache_read=%d out=%d (prompt tokens add up across turns because each turn\n"+
		"           re-sends the history; servers that cache the shared prefix count it as cache_read)\n\n",
		stats.inputTokens, stats.cacheReadTokens, stats.outputTokens)
}

// promptTokens is the full size of the prompt the server processed. With
// prompt caching, input_tokens counts only the part that wasn't cached;
// Ollama, for example, reuses the conversation prefix from the previous
// turn and reports just the newly appended tokens as input_tokens.
func promptTokens(u claude.Usage) int {
	return u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
}

// truncationSuspected compares the payload size with the prompt tokens the
// server says it processed. JSON-heavy text is roughly 3-4 bytes per token;
// far more bytes per token means the server (e.g. Ollama with a small
// context window) most likely dropped part of the prompt.
func truncationSuspected(payloadBytes int, u claude.Usage) (bytesPerToken int, suspected bool) {
	pt := promptTokens(u)
	if pt == 0 {
		return 0, false
	}
	bpt := payloadBytes / pt
	return bpt, bpt > 8
}

// ---- formatting helpers ----

func size(n int) string {
	if n < 1024 {
		return fmt.Sprintf("%d B", n)
	}
	return fmt.Sprintf("%.1f KB", float64(n)/1024)
}

func compact(b []byte) string {
	s := strings.TrimSpace(string(b))
	if s == "" || s == "{}" || s == "null" {
		return ""
	}
	return oneLine(s, 300)
}

func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	return truncate(s, max)
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + fmt.Sprintf("… [+%d more chars]", len(s)-max)
}

func indent(s string) string {
	return "    │ " + strings.ReplaceAll(strings.TrimRight(s, "\n"), "\n", "\n    │ ")
}
