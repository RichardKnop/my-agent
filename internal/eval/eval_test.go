package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/richardknop/my-agent/internal/agent"
	"github.com/richardknop/my-agent/internal/claude"
	"github.com/richardknop/my-agent/internal/data"
)

var payments = Expect{
	RootCauseService:   "payments",
	TriggeringChange:   &Change{Service: "payments", Version: "v2.14.0"},
	AffectedServices:   []string{"payments", "checkout", "frontend"},
	UnaffectedServices: []string{"inventory", "inventory-db"},
}

func TestGradeReport(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		expect  Expect
		deploys []data.Deploy // testDeploys if nil
		report  *agent.Report
		want    Grade
	}{
		{
			name:   "exact answer",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments", TriggeringChange: "v2.14.0",
				AffectedServices: []string{"payments", "checkout", "frontend"}},
			want: Grade{RootCauseService: true, TriggeringChange: true, AffectedRecall: 1, Pass: true},
		},
		{
			// The phrasings qwen3-coder actually used.
			name:   "free phrasing and case",
			expect: payments,
			report: &agent.Report{RootCauseService: " Payments ", TriggeringChange: "V2.14.0 deploy to payments service",
				AffectedServices: []string{"Checkout", "payments"}},
			want: Grade{RootCauseService: true, TriggeringChange: true, AffectedRecall: 2.0 / 3, Pass: true},
		},
		{
			name:   "right change, wrong service",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments-db", TriggeringChange: "payments v2.14.0"},
			want:   Grade{TriggeringChange: true},
		},
		{
			name:   "bare version: root_cause_service says which service",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments", TriggeringChange: "v2.14.0"},
			want:   Grade{RootCauseService: true, TriggeringChange: true, Pass: true},
		},
		{
			// Only payments shipped v2.14.0, so the version alone names the
			// right deploy; the wrong root cause is graded separately.
			name:   "bare version names the deploy even with the wrong root-cause service",
			expect: payments,
			report: &agent.Report{RootCauseService: "checkout", TriggeringChange: "v2.14.0"},
			want:   Grade{TriggeringChange: true},
		},
		{
			name:    "shared version number: root_cause_service says which service",
			expect:  payments,
			deploys: sharedVersionDeploys,
			report:  &agent.Report{RootCauseService: "payments", TriggeringChange: "v2.14.0"},
			want:    Grade{RootCauseService: true, TriggeringChange: true, Pass: true},
		},
		{
			name:    "shared version number with the wrong root-cause service",
			expect:  payments,
			deploys: sharedVersionDeploys,
			report:  &agent.Report{RootCauseService: "checkout", TriggeringChange: "v2.14.0"},
			want:    Grade{},
		},
		{
			name:   "right version attributed to another service",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments", TriggeringChange: "checkout v2.14.0"},
			want:   Grade{RootCauseService: true},
		},
		{
			// From a real eval run: payments-db never deploys, so naming it
			// describes what the change did, not who shipped it.
			name:   "a service the change configured is context, not attribution",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments",
				TriggeringChange: "v2.14.0 deployment reducing payments-db connection pool timeout from 5000ms to 500ms"},
			want: Grade{RootCauseService: true, TriggeringChange: true, Pass: true},
		},
		{
			name:    "payments-db is not payments when it deploys too",
			expect:  payments,
			deploys: append([]data.Deploy{{Service: "payments-db", Version: "v9.0.0"}}, testDeploys...),
			report:  &agent.Report{RootCauseService: "payments", TriggeringChange: "payments-db v2.14.0 migration"},
			want:    Grade{RootCauseService: true},
		},
		{
			name:   "singular service name, other services mentioned too",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments",
				TriggeringChange: "the v2.14.0 deploy to the Payment service, which broke checkout"},
			want: Grade{RootCauseService: true, TriggeringChange: true, Pass: true},
		},
		{
			name:   "right service, no change named",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments"},
			want:   Grade{RootCauseService: true},
		},
		{
			name:   "the rollback is not the trigger",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments", TriggeringChange: "v2.14.1"},
			want:   Grade{RootCauseService: true},
		},
		{
			name:   "control services named as affected",
			expect: payments,
			report: &agent.Report{RootCauseService: "payments", TriggeringChange: "v2.14.0",
				AffectedServices: []string{"payments", "inventory", "inventory-db"}},
			want: Grade{RootCauseService: true, TriggeringChange: true, AffectedRecall: 1.0 / 3,
				WronglyAffected: []string{"inventory", "inventory-db"}, Pass: true},
		},
		{
			name:   "no triggering change expected and none given",
			expect: Expect{RootCauseService: "payments-db"},
			report: &agent.Report{RootCauseService: "payments-db", TriggeringChange: " "},
			want:   Grade{RootCauseService: true, TriggeringChange: true, AffectedRecall: 1, Pass: true},
		},
		{
			name:   "no triggering change expected but one blamed",
			expect: Expect{RootCauseService: "payments-db"},
			report: &agent.Report{RootCauseService: "payments-db", TriggeringChange: "v2.14.0"},
			want:   Grade{RootCauseService: true, AffectedRecall: 1},
		},
		{
			name:   "failed run",
			expect: payments,
			report: nil,
			want:   Grade{},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			deploys := tc.deploys
			if deploys == nil {
				deploys = testDeploys
			}
			got := GradeReport(tc.expect, deploys, tc.report)
			assert.InDelta(t, tc.want.AffectedRecall, got.AffectedRecall, 1e-9)
			got.AffectedRecall = tc.want.AffectedRecall
			assert.Equal(t, tc.want, got)
		})
	}
}

// testDeploys mirrors the committed dataset's deploy history.
var testDeploys = []data.Deploy{
	{Service: "checkout", Version: "v4.2.0"},
	{Service: "payments", Version: "v2.13.2"},
	{Service: "payments", Version: "v2.14.0"},
	{Service: "payments", Version: "v2.14.1"},
}

// sharedVersionDeploys has two services that shipped the same version
// number, so the version alone doesn't say which deploy is meant.
var sharedVersionDeploys = append([]data.Deploy{{Service: "checkout", Version: "v2.14.0"}}, testDeploys...)

// writeScenario creates a scenario directory with an empty data directory.
func writeScenario(t *testing.T, root, name, scenarioJSON string) string {
	t.Helper()
	dir := filepath.Join(root, name)
	require.NoError(t, os.MkdirAll(filepath.Join(dir, "data"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "scenario.json"), []byte(scenarioJSON), 0o644))
	return dir
}

const minimalScenario = `{"now": "2026-08-24T09:00:00Z", "questions": ["why?"], "expect": {"root_cause_service": "payments"}}`

func TestLoadScenario(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		json    string
		wantErr string
	}{
		{name: "valid", json: minimalScenario},
		{
			name:    "misspelled expectation",
			json:    `{"now": "2026-08-24T09:00:00Z", "questions": ["why?"], "expect": {"root_cause": "payments"}}`,
			wantErr: `unknown field "root_cause"`,
		},
		{
			name:    "triggering change without a version",
			json:    `{"now": "2026-08-24T09:00:00Z", "questions": ["why?"], "expect": {"triggering_change": {"service": "payments"}}}`,
			wantErr: "triggering_change needs both service and version",
		},
		{
			name:    "no questions",
			json:    `{"now": "2026-08-24T09:00:00Z", "expect": {}}`,
			wantErr: "no questions",
		},
		{
			name:    "bad now",
			json:    `{"now": "yesterday", "questions": ["why?"]}`,
			wantErr: `invalid now "yesterday"`,
		},
		{
			name:    "missing data directory",
			json:    `{"now": "2026-08-24T09:00:00Z", "questions": ["why?"], "data_dir": "nope"}`,
			wantErr: "data directory",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			dir := writeScenario(t, t.TempDir(), "sc", tc.json)

			sc, err := LoadScenario(dir)

			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				assert.Contains(t, err.Error(), "scenario sc", "errors name the scenario")
				return
			}
			require.NoError(t, err)
			assert.Equal(t, "sc", sc.Name)
			assert.Equal(t, filepath.Join(dir, "data"), sc.DataPath())
		})
	}
}

func TestLoadScenariosFiltersAndSkipsOtherDirectories(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeScenario(t, root, "zeta-incident", minimalScenario)
	writeScenario(t, root, "alpha-incident", minimalScenario)
	writeScenario(t, root, "quiet-day", minimalScenario)
	require.NoError(t, os.MkdirAll(filepath.Join(root, "results"), 0o755))

	all, err := LoadScenarios(root, "")
	require.NoError(t, err)
	var names []string
	for _, sc := range all {
		names = append(names, sc.Name)
	}
	assert.Equal(t, []string{"alpha-incident", "quiet-day", "zeta-incident"}, names, "sorted, results/ skipped")

	some, err := LoadScenarios(root, "incident")
	require.NoError(t, err)
	assert.Len(t, some, 2)

	_, err = LoadScenarios(root, "nothing")
	assert.ErrorContains(t, err, `no scenarios under`)
}

// The committed scenarios must load: a broken scenario.json should fail CI,
// not the next (possibly slow, possibly paid) eval run.
func TestCommittedScenariosAreValid(t *testing.T) {
	t.Parallel()
	scenarios, err := LoadScenarios(filepath.Join("..", "..", "evals"), "")
	require.NoError(t, err)
	for _, sc := range scenarios {
		assert.NotEmpty(t, sc.Description, sc.Name)
		assert.NotEmpty(t, sc.Expect.RootCauseService, sc.Name)
	}
}

func TestReadTrace(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "run.jsonl")
	lines := `{"event":"run_start","elapsed_ms":0}
{"event":"llm_response","text":"<function=query_logs>","text_tool_call":true}
{"event":"llm_response"}
{"event":"tool_call","tool":"query_logs","is_error":false}
{"event":"tool_call","tool":"query_logs","is_error":true}
{"event":"tool_call","tool":"get_deploys","is_error":false}
{"event":"tool_call","tool":"submit_report","is_error":true}
{"event":"tool_call","tool":"submit_report","is_error":false}
{"event":"run_end","elapsed_ms":4200,"turns":5,"total_input_tokens":900,"total_cache_read_input_tokens":3000,"total_output_tokens":250,"budget_exhausted":true,"report":{"summary":"x"}}
`
	require.NoError(t, os.WriteFile(path, []byte(lines), 0o644))

	stats, err := ReadTrace(path)

	require.NoError(t, err)
	assert.Equal(t, RunStats{
		Turns: 5, InputTokens: 900, CacheReadTokens: 3000, OutputTokens: 250,
		BudgetExhausted: true, DurationMS: 4200,
		ToolCalls:       map[string]int{"query_logs": 2, "get_deploys": 1},
		ToolErrors:      1,
		RejectedReports: 1,
		TextToolCalls:   1,
	}, stats)
}

func TestReadTraceRequiresRunEnd(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "run.jsonl")
	require.NoError(t, os.WriteFile(path, []byte(`{"event":"run_start"}`+"\n"), 0o644))

	_, err := ReadTrace(path)

	assert.ErrorContains(t, err, "no run_end event")
}

// scriptedLLM replays replies in order across all runs.
type scriptedLLM struct {
	t       *testing.T
	replies []scriptedReply
	calls   int
}

type scriptedReply struct {
	resp *claude.Response
	err  error
}

func (s *scriptedLLM) CreateMessage(context.Context, claude.Request) (*claude.Response, error) {
	s.calls++
	if s.calls > len(s.replies) {
		s.t.Fatalf("unexpected LLM call #%d", s.calls)
	}
	r := s.replies[s.calls-1]
	return r.resp, r.err
}

func toolUse(id, name, input string) scriptedReply {
	return scriptedReply{resp: &claude.Response{
		StopReason: "tool_use",
		Content:    []claude.ContentBlock{{Type: "tool_use", ID: id, Name: name, Input: json.RawMessage(input)}},
		Usage:      claude.Usage{InputTokens: 100, OutputTokens: 10},
	}}
}

// The runner end to end, minus the model: the committed scenario and
// dataset, the real loop and tools, real traces. Two questions: one run
// investigates and submits the right answer, the other submits an invalid
// report, then never answers properly and runs out of turns.
func TestRunnerRunsGradesAndMeasuresEachQuestion(t *testing.T) {
	t.Parallel()

	sc, err := LoadScenario(filepath.Join("..", "..", "evals", "payments-pool-timeout"))
	require.NoError(t, err)
	sc.Questions = sc.Questions[:2]

	report := `{"summary": "pool timeout", "root_cause_service": "payments",
		"triggering_change": "payments v2.14.0", "affected_services": ["payments", "checkout"],
		"evidence": ["deploy at 14:31"], "confidence": "high", "confidence_reason": "lines up",
		"recommended_action": "roll back"}`
	llm := &scriptedLLM{t: t, replies: []scriptedReply{
		// question 0: one tool call, then the report
		toolUse("c1", "get_deploys", `{"start": "2026-08-23T14:00:00Z", "end": "2026-08-23T16:00:00Z"}`),
		toolUse("c2", agent.SubmitReportToolName, report),
		// question 1: an invalid report, then a tool call on the final turn
		toolUse("c3", agent.SubmitReportToolName, `{"summary": "x", "confidence": "sure"}`),
		toolUse("c4", "list_services", `{}`),
	}}
	cfg := agent.DefaultConfig()
	cfg.MaxTurns = 2
	runner := &Runner{LLM: llm, Config: cfg, TraceDir: t.TempDir()}

	var progress []RunResult
	results, err := runner.RunScenario(context.Background(), sc, 1, func(r RunResult) { progress = append(progress, r) })

	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, results, progress, "onResult sees every result as it finishes")

	ok := results[0]
	assert.Equal(t, 0, ok.Question)
	assert.Equal(t, sc.Questions[0], ok.QuestionText)
	assert.Empty(t, ok.Error)
	require.NotNil(t, ok.Report)
	assert.True(t, ok.Grade.Pass)
	assert.InDelta(t, 2.0/3, ok.Grade.AffectedRecall, 1e-9)
	assert.Equal(t, 2, ok.Stats.Turns)
	assert.Equal(t, 200, ok.Stats.InputTokens)
	assert.Equal(t, map[string]int{"get_deploys": 1}, ok.Stats.ToolCalls)
	assert.Zero(t, ok.Stats.ToolErrors, "the real get_deploys ran against the committed data")
	assert.FileExists(t, ok.TracePath)

	failed := results[1]
	assert.Contains(t, failed.Error, "exceeded max turns (2)")
	assert.Nil(t, failed.Report)
	assert.False(t, failed.Grade.Pass)
	assert.Equal(t, 2, failed.Stats.Turns, "a failed run is still measured from its trace")
	assert.Equal(t, 1, failed.Stats.RejectedReports)
	assert.False(t, failed.Stats.ModelError, "an agent failure, not an infrastructure one")

	sum := Summarize(sc.Name, results)
	assert.Equal(t, 2, sum.Runs)
	assert.Equal(t, 1, sum.Passed)
	assert.Equal(t, 1, sum.Errors)
	assert.Equal(t, 1, sum.RejectedReports)
	assert.InDelta(t, 2, sum.AvgTurns, 1e-9)
	assert.InDelta(t, 1.0/3, sum.AffectedRecall, 1e-9, "failed runs count as zero recall")

	var table bytes.Buffer
	require.NoError(t, WriteTable(&table, []Summary{sum}))
	assert.Contains(t, table.String(), "payments-pool-timeout")
	assert.Contains(t, table.String(), "1/2")
}

// A model API failure is an outage, not an agent mistake: the eval stops
// instead of recording every remaining run as a failure.
func TestRunnerStopsWhenTheModelIsUnavailable(t *testing.T) {
	t.Parallel()

	sc, err := LoadScenario(filepath.Join("..", "..", "evals", "payments-pool-timeout"))
	require.NoError(t, err)
	llm := &scriptedLLM{t: t, replies: []scriptedReply{
		{err: errors.New("dial tcp [::1]:11434: connect: connection refused")},
	}}
	runner := &Runner{LLM: llm, Config: agent.DefaultConfig(), TraceDir: t.TempDir()}

	results, err := runner.RunScenario(context.Background(), sc, 3, nil)

	require.ErrorIs(t, err, ErrModelUnavailable)
	assert.Contains(t, err.Error(), "connection refused")
	require.Len(t, results, 1, "no runs after the failing one")
	assert.True(t, results[0].Stats.ModelError)
	assert.Equal(t, 1, llm.calls)
}
