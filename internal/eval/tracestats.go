package eval

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"github.com/richardknop/my-agent/internal/agent"
)

// RunStats are the process measurements of one run, read from its trace.
// They answer "how did it get there, and at what cost", which a correct
// answer alone doesn't: a prompt change that keeps the pass rate but doubles
// the turns is still a regression.
type RunStats struct {
	Turns           int  `json:"turns"`
	InputTokens     int  `json:"input_tokens"`
	CacheReadTokens int  `json:"cache_read_input_tokens"`
	OutputTokens    int  `json:"output_tokens"`
	BudgetExhausted bool `json:"budget_exhausted"`
	DurationMS      int  `json:"duration_ms"`
	// ToolCalls counts investigation tool calls by name (submit_report is
	// counted separately in RejectedReports).
	ToolCalls map[string]int `json:"tool_calls"`
	// ToolErrors counts investigation tool calls that returned an error,
	// e.g. a malformed timestamp.
	ToolErrors int `json:"tool_errors"`
	// RejectedReports counts submit_report calls the loop sent back as
	// invalid. Each one cost a turn.
	RejectedReports int `json:"rejected_reports"`
	// TextToolCalls counts responses where the model wrote a tool call as
	// text that the server didn't parse, so nothing ran.
	TextToolCalls int `json:"text_tool_calls"`
	// ModelError is set when the run failed because the model API call
	// failed (after the client's retries): an infrastructure problem, not
	// something the agent got wrong.
	ModelError bool `json:"model_error,omitempty"`
}

// ReadTrace reads a run's JSONL trace. Reading the trace rather than asking
// the agent for its counters is deliberate: evals then depend only on the
// trace format, the same record a production pipeline (stage 5) would
// consume, and they double as a check that the trace has what's needed.
func ReadTrace(path string) (RunStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return RunStats{}, err
	}
	defer f.Close()

	stats := RunStats{ToolCalls: map[string]int{}}
	sawEnd := false
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 10*1024*1024) // run_end carries the whole report
	for sc.Scan() {
		var ev struct {
			Event     string `json:"event"`
			ElapsedMS int    `json:"elapsed_ms"`
			// tool_call
			Tool    string `json:"tool"`
			IsError bool   `json:"is_error"`
			// llm_response
			TextToolCall bool `json:"text_tool_call"`
			// run_end
			Turns           int  `json:"turns"`
			InputTokens     int  `json:"total_input_tokens"`
			CacheReadTokens int  `json:"total_cache_read_input_tokens"`
			OutputTokens    int  `json:"total_output_tokens"`
			BudgetExhausted bool `json:"budget_exhausted"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			return RunStats{}, fmt.Errorf("%s: %w", path, err)
		}
		switch ev.Event {
		case "tool_call":
			if ev.Tool == agent.SubmitReportToolName {
				if ev.IsError {
					stats.RejectedReports++
				}
				continue
			}
			stats.ToolCalls[ev.Tool]++
			if ev.IsError {
				stats.ToolErrors++
			}
		case "llm_response":
			if ev.TextToolCall {
				stats.TextToolCalls++
			}
		case "llm_error":
			stats.ModelError = true
		case "run_end":
			sawEnd = true
			stats.Turns = ev.Turns
			stats.InputTokens = ev.InputTokens
			stats.CacheReadTokens = ev.CacheReadTokens
			stats.OutputTokens = ev.OutputTokens
			stats.BudgetExhausted = ev.BudgetExhausted
			stats.DurationMS = ev.ElapsedMS
		}
	}
	if err := sc.Err(); err != nil {
		return RunStats{}, fmt.Errorf("%s: %w", path, err)
	}
	if !sawEnd {
		return RunStats{}, fmt.Errorf("%s: no run_end event", path)
	}
	return stats, nil
}
