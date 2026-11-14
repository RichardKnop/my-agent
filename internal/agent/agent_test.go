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

func TestInvestigate_FinalAnswerWithoutTools(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("end_turn", textBlock("Root cause: nothing is broken.")),
	}}
	a := newTestAgent(t, llm, 8)

	answer, err := a.Investigate(context.Background(), "is anything broken?")

	require.NoError(t, err)
	assert.Equal(t, "Root cause: nothing is broken.", answer)
	require.Len(t, llm.requests, 1)

	req := llm.requests[0]
	assert.Equal(t, DefaultConfig().Model, req.Model)
	assert.Contains(t, req.System, "at most 8 turns", "the system prompt must state the turn budget")
	assert.Nil(t, req.ToolChoice, "tool use is left to the model before the final turn")
	assert.Contains(t, req.System, "Current time: 2026-08-24T09:00:00Z (Monday)",
		"the system prompt must tell the model what 'now' is")
	// Every registry definition goes on the wire unchanged, converted to the
	// Messages API's tool type.
	defs := tools.NewRegistry(testStore()).Definitions()
	require.Len(t, req.Tools, len(defs))
	for i, d := range defs {
		assert.Equal(t, claude.Tool{Name: d.Name, Description: d.Description, InputSchema: d.InputSchema}, req.Tools[i])
	}
	require.Len(t, req.Messages, 1)
	assert.Equal(t, "user", req.Messages[0].Role)
	assert.Equal(t, "is anything broken?", req.Messages[0].Content[0].Text)
}

func TestInvestigate_ToolCallThenAnswer(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use",
			textBlock("Let me look at the services."),
			toolUse("call_1", "list_services", `{}`)),
		reply("end_turn", textBlock("payments is the culprit")),
	}}
	a := newTestAgent(t, llm, 8)

	answer, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.Equal(t, "payments is the culprit", answer)
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
		reply("end_turn", textBlock("done")),
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
		reply("end_turn", textBlock("recovered")),
	}}
	a := newTestAgent(t, llm, 8)

	answer, err := a.Investigate(context.Background(), "why?")

	// A failing tool is not a failed run: the error goes back as an
	// is_error tool_result so the model can correct itself.
	require.NoError(t, err)
	assert.Equal(t, "recovered", answer)

	results := llm.requests[1].Messages[2].Content
	require.Len(t, results, 2)
	assert.True(t, results[0].IsError)
	assert.Contains(t, results[0].Content, `unknown tool "no_such_tool"`)
	assert.True(t, results[1].IsError)
	assert.Contains(t, results[1].Content, "bad input")
}

func TestInvestigate_StopsAtMaxTurns(t *testing.T) {
	t.Parallel()

	loop := reply("tool_use", toolUse("call", "list_services", `{}`))
	llm := &fakeLLM{replies: []fakeReply{loop, loop, loop}}
	a := newTestAgent(t, llm, 3)

	answer, err := a.Investigate(context.Background(), "why?")

	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeded max turns (3)")
	assert.Contains(t, err.Error(), "kept calling tools on the final turn")
	assert.Empty(t, answer)
	assert.Len(t, llm.requests, 3, "no LLM call beyond the turn cap")
}

func TestInvestigate_LLMErrorAbortsRun(t *testing.T) {
	t.Parallel()

	apiErr := errors.New("messages api error (status 400): bad request")
	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use", toolUse("call_1", "list_services", `{}`)),
		{err: apiErr},
	}}
	a := newTestAgent(t, llm, 8)

	answer, err := a.Investigate(context.Background(), "why?")

	require.ErrorIs(t, err, apiErr, "the client error is wrapped, not replaced")
	assert.Contains(t, err.Error(), "turn 2")
	assert.Empty(t, answer)
}

func TestInvestigate_FinalAnswerJoinsTextBlocksOnly(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("end_turn", textBlock("Root cause: X.\n"), textBlock("Confidence: high")),
	}}
	a := newTestAgent(t, llm, 8)

	answer, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.Equal(t, "Root cause: X.\nConfidence: high", answer)
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
// one run_end carrying the turn count, token totals and, on failure, the
// error. Anything reading traces (stage 5) can rely on that.
func TestInvestigate_EveryRunEndsWithOneRunEndEvent(t *testing.T) {
	t.Parallel()

	loop := reply("tool_use", toolUse("call", "list_services", `{}`))
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
			name:      "final answer",
			replies:   []fakeReply{loop, reply("end_turn", textBlock("done"))},
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
			name:      "answer forced on the final turn",
			replies:   []fakeReply{loop, reply("end_turn", textBlock("partial answer"))},
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
			} else {
				require.Error(t, err)
				assert.Contains(t, end["error"], tc.wantError)
				assert.Equal(t, err.Error(), end["error"], "trace records the same error the caller gets")
			}
		})
	}
}

// The loop treats its turn cap as a budget the model knows about: before the
// final turn it tells the model to answer, and on the final turn it forbids
// tools, so a long investigation ends with a (partial) report instead of an
// error.
func TestInvestigate_FinalTurnForcesAnAnswer(t *testing.T) {
	t.Parallel()

	llm := &fakeLLM{replies: []fakeReply{
		reply("tool_use", toolUse("call_1", "list_services", `{}`)),
		reply("tool_use", toolUse("call_2", "list_services", `{}`)),
		reply("end_turn", textBlock("Root cause: probably payments. Confidence: low")),
	}}
	a := newTestAgent(t, llm, 3)

	answer, err := a.Investigate(context.Background(), "why?")

	require.NoError(t, err)
	assert.Equal(t, "Root cause: probably payments. Confidence: low", answer)
	require.Len(t, llm.requests, 3)

	assert.Nil(t, llm.requests[0].ToolChoice)
	assert.Nil(t, llm.requests[1].ToolChoice)
	assert.Equal(t, &claude.ToolChoice{Type: "none"}, llm.requests[2].ToolChoice)
	assert.NotEmpty(t, llm.requests[2].Tools,
		"tools are still defined on the final turn: the history contains tool_use blocks")

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
