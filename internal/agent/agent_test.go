package agent

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/richardknop/my-agent/internal/claude"
	"github.com/richardknop/my-agent/internal/data"
	"github.com/richardknop/my-agent/internal/tools"
	"github.com/richardknop/my-agent/internal/trace"
)

// These tests drive the agent loop with a scripted fake model instead of a
// real one. Everything else is real: the tool registry runs its handlers
// against a tiny in-memory store. The fake records each request, so the
// tests can assert on what the loop actually sent: how the conversation
// grows, whether every tool_use gets a matching tool_result, and so on.

// fakeLLM replays scripted replies in order and records every request.
type fakeLLM struct {
	t        *testing.T
	replies  []fakeReply
	requests []claude.Request
}

type fakeReply struct {
	resp *claude.Response
	err  error
}

func (f *fakeLLM) CreateMessage(_ context.Context, req claude.Request) (*claude.Response, error) {
	f.requests = append(f.requests, req)
	if len(f.requests) > len(f.replies) {
		f.t.Fatalf("unexpected LLM call #%d: only %d replies scripted", len(f.requests), len(f.replies))
	}
	r := f.replies[len(f.requests)-1]
	return r.resp, r.err
}

// ---- helpers for building scripted replies ----

func textBlock(s string) claude.ContentBlock {
	return claude.ContentBlock{Type: "text", Text: s}
}

func toolUse(id, name, input string) claude.ContentBlock {
	return claude.ContentBlock{Type: "tool_use", ID: id, Name: name, Input: json.RawMessage(input)}
}

func reply(stopReason string, blocks ...claude.ContentBlock) fakeReply {
	return fakeReply{resp: &claude.Response{
		Role:       "assistant",
		Content:    blocks,
		StopReason: stopReason,
		Usage:      claude.Usage{InputTokens: 100, OutputTokens: 10},
	}}
}

// validReport is a well-formed submit_report input.
const validReport = `{
	"summary": "payments rejected connections after a config change",
	"root_cause_service": "payments",
	"triggering_change": "v9.9.9",
	"affected_services": ["payments", "checkout"],
	"evidence": ["checkout error_rate 0.4 at 14:35"],
	"confidence": "high",
	"confidence_reason": "deploy lines up with the errors",
	"recommended_action": "roll back"
}`

func submit(id, input string) claude.ContentBlock {
	return toolUse(id, submitReportToolName, input)
}

var testNow = time.Date(2026, 8, 24, 9, 0, 0, 0, time.UTC)

func testStore() *data.Store {
	return &data.Store{
		Services: []data.Service{
			{Name: "checkout", Description: "checkout flow", DependsOn: []string{"payments"}},
			{Name: "payments", Description: "charges cards"},
		},
		Metrics: map[string]map[string][]data.MetricPoint{},
	}
}

func newTestAgent(t *testing.T, llm *fakeLLM, maxTurns int) *Agent {
	t.Helper()
	llm.t = t
	cfg := DefaultConfig()
	cfg.MaxTurns = maxTurns
	cfg.Now = testNow
	// nil tracer and nil LogWriter are both no-ops.
	return New(llm, tools.NewRegistry(testStore()), nil, cfg)
}

// ---- tests ----

func TestInvestigate_ReportWithoutTools(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use", submit("call_1", validReport)),
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "is anything broken?")

	require.NoError(t, err)
	assert.Equal(t, &Report{
		Summary:           "payments rejected connections after a config change",
		RootCauseService:  "payments",
		TriggeringChange:  "v9.9.9",
		AffectedServices:  []string{"payments", "checkout"},
		Evidence:          []string{"checkout error_rate 0.4 at 14:35"},
		Confidence:        "high",
		ConfidenceReason:  "deploy lines up with the errors",
		RecommendedAction: "roll back",
	}, report)
	require.Len(t, llm.requests, 1)

	req := llm.requests[0]
	assert.Equal(t, DefaultConfig().Model, req.Model)
	assert.Contains(t, req.System, "at most 8 turns", "the system prompt must state the turn budget")
	assert.Contains(t, req.System, "submit_report", "the system prompt must say how to answer")
	assert.Nil(t, req.ToolChoice, "tool use is left to the model before the final turn")
	assert.Contains(t, req.System, "Current time: 2026-08-24T09:00:00Z (Monday)",
		"the system prompt must tell the model what 'now' is")
	// Every registry definition goes on the wire unchanged, converted to the
	// Messages API's tool type, followed by the loop's own submit_report.
	defs := tools.NewRegistry(testStore()).Definitions()
	require.Len(t, req.Tools, len(defs)+1)
	for i, d := range defs {
		assert.Equal(t, claude.Tool{Name: d.Name, Description: d.Description, InputSchema: d.InputSchema}, req.Tools[i])
	}
	assert.Equal(t, submitReportTool, req.Tools[len(defs)])
	require.Len(t, req.Messages, 1)
	assert.Equal(t, "user", req.Messages[0].Role)
	assert.Equal(t, "is anything broken?", req.Messages[0].Content[0].Text)
}

func TestInvestigate_ToolCallThenReport(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			textBlock("Let me look at the services."),
			toolUse("call_1", "list_services", `{}`)),
		reply("tool_use", submit("call_2", validReport)),
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.Equal(t, "payments", report.RootCauseService)
	require.Len(t, llm.requests, 2)

	// The model is stateless: turn 2 re-sends the whole conversation, i.e.
	// question, the assistant's tool_use turn verbatim, then our results.
	msgs := llm.requests[1].Messages
	require.Len(t, msgs, 3)
	assert.Equal(t, "user", msgs[0].Role)
	assert.Equal(t, "assistant", msgs[1].Role)
	assert.Equal(t, llm.replies[0].resp.Content, msgs[1].Content, "assistant turn is echoed back unchanged")
	assert.Equal(t, "user", msgs[2].Role, "tool results go back in a user message")

	require.Len(t, msgs[2].Content, 1)
	result := msgs[2].Content[0]
	assert.Equal(t, "tool_result", result.Type)
	assert.Equal(t, "call_1", result.ToolUseID, "tool_result must reference the tool_use id")
	assert.False(t, result.IsError)

	var services []data.Service
	require.NoError(t, json.Unmarshal([]byte(result.Content), &services))
	assert.Len(t, services, 2, "the real list_services handler ran against the store")
}

func TestInvestigate_ParallelToolCallsAnsweredInOneMessage(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			toolUse("call_a", "list_services", `{}`),
			toolUse("call_b", "get_deploys", `{"start":"2026-08-23T00:00:00Z","end":"2026-08-24T00:00:00Z"}`)),
		reply("tool_use", submit("call_c", validReport)),
	}}
	a := newTestAgent(t, llm, 8)

	_, err := a.Investigate(context.Background(), "why?")
	require.NoError(t, err)

	// All results for one assistant turn go back together, in order, in a
	// single user message; the API rejects a tool_use left unanswered.
	msgs := llm.requests[1].Messages
	require.Len(t, msgs, 3)
	results := msgs[2].Content
	require.Len(t, results, 2)
	assert.Equal(t, "call_a", results[0].ToolUseID)
	assert.Equal(t, "call_b", results[1].ToolUseID)
}

func TestInvestigate_ToolErrorsAreFedBackToTheModel(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			toolUse("call_1", "no_such_tool", `{}`),
			toolUse("call_2", "query_logs", `{"start": 42}`)), // "start" is a string, 42 will cause an error
		reply("tool_use", submit("call_3", validReport)),
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	// A failing tool is not a failed run: the error goes back as an
	// is_error tool_result so the model can correct itself.
	require.NoError(t, err)
	assert.NotNil(t, report)

	results := llm.requests[1].Messages[2].Content
	require.Len(t, results, 2)
	assert.True(t, results[0].IsError)
	assert.Contains(t, results[0].Content, `unknown tool "no_such_tool"`)
	assert.True(t, results[1].IsError)
	assert.Contains(t, results[1].Content, "bad input")
}

// An invalid report is a tool error like any other: the model gets told what
// was wrong and resubmits. Other tools asked for alongside it still run.
func TestInvestigate_InvalidReportIsFedBackToTheModel(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			toolUse("call_1", "list_services", `{}`),
			submit("call_2", `{"summary": "", "confidence": "very"}`)),
		reply("tool_use", submit("call_3", validReport)),
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.Equal(t, "payments", report.RootCauseService)

	results := llm.requests[1].Messages[2].Content
	require.Len(t, results, 2)
	assert.Equal(t, "call_1", results[0].ToolUseID)
	assert.False(t, results[0].IsError)
	assert.Equal(t, "call_2", results[1].ToolUseID)
	assert.True(t, results[1].IsError)
	assert.Contains(t, results[1].Content, "summary is required")
	assert.Contains(t, results[1].Content, `confidence must be one of low, medium, high (got "very")`)
}

// More than one submit_report in a response is a mistake, like an invalid
// report: none is accepted (even valid ones), every call gets the same tool
// error, and the model resubmits. Other tools in the response still run.
func TestInvestigate_MultipleReportsInOneResponseAreRejected(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			submit("call_1", validReport),
			toolUse("call_2", "list_services", `{}`),
			submit("call_3", `{"summary": "x", "confidence": "nope"}`)),
		reply("tool_use", submit("call_4", validReport)),
	}}
	a, events := newTracedTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.NotNil(t, report)
	require.Len(t, llm.requests, 2, "no report was accepted on turn 1")

	results := llm.requests[1].Messages[2].Content
	require.Len(t, results, 3, "every tool_use gets a tool_result, in order")
	for i, id := range []string{"call_1", "call_3"} {
		r := results[i*2]
		assert.Equal(t, id, r.ToolUseID)
		assert.True(t, r.IsError)
		assert.Equal(t, "invalid report: submit_report was called 2 times in one response; "+
			"call it exactly once. None of these reports was recorded", r.Content)
	}
	assert.Equal(t, "call_2", results[1].ToolUseID)
	assert.False(t, results[1].IsError, "other tools in the response still run")

	var rejected int
	for _, ev := range events() {
		if ev["event"] == "tool_call" && ev["tool"] == submitReportToolName && ev["is_error"] == true {
			rejected++
		}
	}
	assert.Equal(t, 2, rejected, "each rejected submit_report call is traced")
}

// A valid report ends the run right away; tools requested in the same
// response are not run, since nobody would read their results.
func TestInvestigate_ValidReportEndsTheRunImmediately(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			toolUse("call_1", "no_such_tool", `{}`),
			submit("call_2", validReport)),
	}}
	a, events := newTracedTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.NotNil(t, report)
	assert.Len(t, llm.requests, 1)
	for _, ev := range events() {
		if ev["event"] == "tool_call" {
			assert.Equal(t, submitReportToolName, ev["tool"], "no other tool ran")
		}
	}
}

// A plain-text reply is not an answer. The loop asks again, which costs a
// turn, rather than accept text it can't grade.
func TestInvestigate_PlainTextReplyGetsAReminder(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("end_turn", textBlock("Root cause: payments.")),
		reply("tool_use", submit("call_1", validReport)),
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.NotNil(t, report)
	require.Len(t, llm.requests, 2)

	msgs := llm.requests[1].Messages
	require.Len(t, msgs, 3)
	assert.Equal(t, "assistant", msgs[1].Role)
	assert.Equal(t, "user", msgs[2].Role)
	assert.Equal(t, []claude.ContentBlock{textBlock(submitReminder)}, msgs[2].Content)
}

func TestInvestigate_StopsAtMaxTurns(t *testing.T) {
	t.Parallel()

	loop := reply("tool_use", toolUse("call", "list_services", `{}`))
	llm := &fakeLLM{replies: []fakeReply{loop, loop, loop}}
	a := newTestAgent(t, llm, 3)

	report, err := a.Investigate(context.Background(), "why?")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeded max turns (3) without a report")
	assert.Contains(t, err.Error(), "kept calling tools on the final turn")
	assert.Nil(t, report)
	assert.Len(t, llm.requests, 3, "no LLM call beyond the turn cap")
}

// On the final turn there is no turn left to recover in, so anything but a
// valid report fails the run.
func TestInvestigate_FinalTurnWithoutValidReportFails(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		final   fakeReply
		wantErr string
	}{
		{
			name:    "plain text",
			final:   reply("end_turn", textBlock("Root cause: payments.")),
			wantErr: "exceeded max turns (2) without a report: the model answered in plain text instead of calling submit_report",
		},
		{
			name:    "invalid report",
			final:   reply("tool_use", submit("call_2", `{"summary": "x", "confidence": "sure"}`)),
			wantErr: "turn 2: the model submitted an invalid report on the final turn: invalid report: confidence must be",
		},
		{
			name:    "two reports",
			final:   reply("tool_use", submit("call_2", validReport), submit("call_3", validReport)),
			wantErr: "turn 2: the model submitted an invalid report on the final turn: invalid report: submit_report was called 2 times",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			llm := &fakeLLM{replies: []fakeReply{
				reply("tool_use", toolUse("call_1", "list_services", `{}`)),
				tc.final,
			}}
			a := newTestAgent(t, llm, 2)

			report, err := a.Investigate(context.Background(), "why?")

			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
			assert.Nil(t, report)
		})
	}
}

func TestInvestigate_LLMErrorAbortsRun(t *testing.T) {
	t.Parallel()

	apiErr := errors.New("messages api error (status 400): bad request")
	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use", toolUse("call_1", "list_services", `{}`)),
		{err: apiErr},
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.ErrorIs(t, err, apiErr, "the client error is wrapped, not replaced")
	assert.Contains(t, err.Error(), "turn 2")
	assert.Nil(t, report)
}

// newTracedTestAgent is newTestAgent with a real tracer writing to a temp
// file, plus a function that reads back the trace events written so far.
func newTracedTestAgent(t *testing.T, llm *fakeLLM, maxTurns int) (*Agent, func() []map[string]any) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "run.jsonl")
	tracer, err := trace.New(path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = tracer.Close() })

	a := newTestAgent(t, llm, maxTurns)
	a.tracer = tracer

	events := func() []map[string]any {
		b, err := os.ReadFile(path)
		require.NoError(t, err)
		var out []map[string]any
		for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			var ev map[string]any
			require.NoError(t, json.Unmarshal([]byte(line), &ev))
			out = append(out, ev)
		}
		return out
	}
	return a, events
}

// However a run ends, the trace starts with run_start and ends with exactly
// one run_end carrying the turn count, token totals and either the report or
// the error. Anything reading traces (stage 5) can rely on that.
func TestInvestigate_EveryRunEndsWithOneRunEndEvent(t *testing.T) {
	t.Parallel()

	loop := reply("tool_use", toolUse("call", "list_services", `{}`))
	done := reply("tool_use", submit("call_done", validReport))
	cases := []struct {
		name      string
		replies   []fakeReply
		maxTurns  int
		wantTurns int
		wantIn    int    // total input tokens: 100 per successful response
		wantError string // empty = success
		wantForce bool   // budget_exhausted
	}{
		{
			name:      "report",
			replies:   []fakeReply{loop, done},
			maxTurns:  8,
			wantTurns: 2,
			wantIn:    200,
		},
		{
			name:      "model API error",
			replies:   []fakeReply{loop, {err: errors.New("status 500")}},
			maxTurns:  8,
			wantTurns: 2,
			wantIn:    100,
			wantError: "turn 2: status 500",
		},
		{
			name:      "turn cap",
			replies:   []fakeReply{loop, loop},
			maxTurns:  2,
			wantTurns: 2,
			wantIn:    200,
			wantError: "exceeded max turns (2)",
		},
		{
			name:      "report forced on the final turn",
			replies:   []fakeReply{loop, done},
			maxTurns:  2,
			wantTurns: 2,
			wantIn:    200,
			wantForce: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			llm := &fakeLLM{replies: tc.replies}
			a, events := newTracedTestAgent(t, llm, tc.maxTurns)

			_, err := a.Investigate(context.Background(), "why?")

			evs := events()
			require.NotEmpty(t, evs)
			assert.Equal(t, "run_start", evs[0]["event"])

			var runEnds []map[string]any
			for _, ev := range evs {
				if ev["event"] == "run_end" {
					runEnds = append(runEnds, ev)
				}
			}
			require.Len(t, runEnds, 1, "exactly one run_end per run")
			end := runEnds[0]
			assert.Equal(t, end, evs[len(evs)-1], "run_end is the last event")

			// JSON numbers decode as float64.
			assert.EqualValues(t, tc.wantTurns, end["turns"])
			assert.EqualValues(t, tc.wantIn, end["total_input_tokens"])

			if tc.wantError == "" {
				require.NoError(t, err)
				assert.NotContains(t, end, "error")
				assert.Equal(t, tc.wantForce, end["budget_exhausted"])
				require.Contains(t, end, "report", "run_end records the submitted report")
				assert.Equal(t, "payments", end["report"].(map[string]any)["root_cause_service"])
			} else {
				require.Error(t, err)
				assert.NotContains(t, end, "report")
				assert.Contains(t, end["error"], tc.wantError)
				assert.Equal(t, err.Error(), end["error"], "trace records the same error the caller gets")
			}
		})
	}
}

// The loop treats its turn cap as a budget the model knows about: before the
// final turn it tells the model to submit, and on the final turn it forces
// submit_report, so a long investigation ends with a (partial) report instead
// of an error.
func TestInvestigate_FinalTurnForcesAReport(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use", toolUse("call_1", "list_services", `{}`)),
		reply("tool_use", toolUse("call_2", "list_services", `{}`)),
		reply("tool_use", submit("call_3", validReport)),
	}}
	a := newTestAgent(t, llm, 3)

	report, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.NotNil(t, report)
	require.Len(t, llm.requests, 3)

	assert.Nil(t, llm.requests[0].ToolChoice)
	assert.Nil(t, llm.requests[1].ToolChoice)
	assert.Equal(t, &claude.ToolChoice{Type: "tool", Name: submitReportToolName}, llm.requests[2].ToolChoice)
	assert.Len(t, llm.requests[2].Tools, len(llm.requests[0].Tools),
		"all tools are still defined on the final turn: the history contains tool_use blocks")

	// Turn 2's request: results of turn 1, no note yet.
	turn1Results := llm.requests[1].Messages[2].Content
	require.Len(t, turn1Results, 1)
	assert.Equal(t, "tool_result", turn1Results[0].Type)

	// Final request: results of turn 2, followed by the note in the same
	// user message (tool_result blocks first, as the API requires).
	turn2Results := llm.requests[2].Messages[4].Content
	require.Len(t, turn2Results, 2)
	assert.Equal(t, "tool_result", turn2Results[0].Type)
	assert.Equal(t, "call_2", turn2Results[0].ToolUseID)
	assert.Equal(t, "text", turn2Results[1].Type)
	assert.Equal(t, finalTurnNote, turn2Results[1].Text)
}

// A response cut off by max_tokens is half an answer. The loop fails the
// run rather than return it as if it were the finished report.
func TestInvestigate_CutOffResponseFailsTheRun(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("max_tokens", submit("call_1", `{"summary": "a deploy of pay`)),
	}}
	a := newTestAgent(t, llm, 8)

	report, err := a.Investigate(context.Background(), "why?")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "turn 1: response cut off at max_tokens (4096)")
	assert.Nil(t, report)
	assert.Len(t, llm.requests, 1, "no further turns after a cut-off response")
}
