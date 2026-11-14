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
	l.printf("← LLM response: %s, stop_reason=%s, tokens in=%d out=%d\n",
		latency.Round(10*time.Millisecond), resp.StopReason, resp.Usage.InputTokens, resp.Usage.OutputTokens)

	// JSON-heavy text is roughly 3-4 bytes per token. If far fewer tokens
	// were counted than the payload size implies, the server (e.g. Ollama
	// with a small context window) most likely dropped part of the prompt.
	if in := resp.Usage.InputTokens; in > 0 && payloadBytes/in > 8 {
		l.printf("  ! %s sent but only %d input tokens counted (~%d bytes/token): the server may have\n"+
			"    truncated the context. With Ollama, raise OLLAMA_CONTEXT_LENGTH.\n",
			size(payloadBytes), in, payloadBytes/in)
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

func (l *stepLog) runEnd(turns, totalIn, totalOut int, err error) {
	if !l.on(1) {
		return
	}
	l.printf("\n━━━━ run end ━━━━\n")
	if err != nil {
		l.printf("result   : FAILED: %v\n", err)
	} else {
		l.printf("result   : final answer after %d turn(s)\n", turns)
	}
	l.printf("tokens   : in=%d out=%d (input adds up across turns because each turn re-sends the history)\n\n", totalIn, totalOut)
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
