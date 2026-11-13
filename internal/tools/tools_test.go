package tools

import (
	"encoding/json"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/richardknop/my-agent/internal/data"
)

// loadTestStore loads the real generated dataset so these tests exercise
// the tools against the same data the agent sees. Run scripts/gen_data.py
// first if data/*.json don't exist yet.
func loadTestStore(t *testing.T) *data.Store {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	dataDir := filepath.Join(filepath.Dir(thisFile), "..", "..", "data")
	store, err := data.Load(dataDir)
	require.NoError(t, err, "loading dataset (did you run `python3 scripts/gen_data.py`?)")
	return store
}

// incidentDay returns the dataset's incident day ("YYYY-MM-DD"), read from
// the payments v2.14.0 deploy, since gen_data.py generates the scenario
// relative to the day it's run rather than on a fixed date.
func incidentDay(t *testing.T, store *data.Store) string {
	t.Helper()
	for _, d := range store.Deploys {
		if d.Service == "payments" && d.Version == "v2.14.0" {
			return d.Timestamp[:10]
		}
	}
	t.Fatal("payments v2.14.0 deploy not found in dataset")
	return ""
}

// call runs a tool through the registry with input marshalled from args.
func call(t *testing.T, store *data.Store, tool string, args any) (string, error) {
	t.Helper()
	input, err := json.Marshal(args)
	require.NoError(t, err)
	return NewRegistry().Call(tool, store, input)
}

// mustCall is call for the happy path: the tool must succeed, and its JSON
// result is decoded into out.
func mustCall(t *testing.T, store *data.Store, tool string, args, out any) {
	t.Helper()
	res, err := call(t, store, tool, args)
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal([]byte(res), out))
}

type groupedLogsResult struct {
	Mode           string     `json:"mode"`
	TotalMatches   int        `json:"total_matches"`
	TotalGroups    int        `json:"total_groups"`
	ReturnedGroups int        `json:"returned_groups"`
	Truncated      bool       `json:"truncated"`
	Groups         []LogGroup `json:"groups"`
}

type rawLogsResult struct {
	Mode         string          `json:"mode"`
	TotalMatches int             `json:"total_matches"`
	Returned     int             `json:"returned"`
	Truncated    bool            `json:"truncated"`
	Entries      []data.LogEntry `json:"entries"`
}

func TestListServices(t *testing.T) {
	store := loadTestStore(t)

	var services []data.Service
	mustCall(t, store, "list_services", map[string]string{}, &services)

	names := map[string]bool{}
	for _, s := range services {
		names[s.Name] = true
	}
	for _, want := range []string{"frontend", "checkout", "payments", "payments-db"} {
		assert.True(t, names[want], "expected service %q in catalog", want)
	}
}

func TestQueryMetricsFindsIncidentSpike(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)

	var result struct {
		Summary struct {
			Max float64 `json:"max"`
		} `json:"summary"`
	}
	mustCall(t, store, "query_metrics", map[string]string{
		"service": "payments",
		"metric":  "error_rate",
		"start":   day + "T14:00:00Z",
		"end":     day + "T15:00:00Z",
	}, &result)

	// Baseline error_rate for payments is ~0.2%; the incident should push
	// the max well past that within this window.
	assert.GreaterOrEqual(t, result.Summary.Max, 5.0,
		"expected a clear error_rate spike for payments in the incident window")
}

func TestQueryMetricsUnknownServiceErrors(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)

	_, err := call(t, store, "query_metrics", map[string]string{
		"service": "does-not-exist",
		"metric":  "error_rate",
		"start":   day + "T14:00:00Z",
		"end":     day + "T15:00:00Z",
	})
	assert.ErrorContains(t, err, `unknown service "does-not-exist"`)
}

func TestQueryLogsGroupedByDefault(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)

	var res groupedLogsResult
	mustCall(t, store, "query_logs", map[string]string{
		"service": "payments",
		"level":   "ERROR",
		"start":   day + "T14:00:00Z",
		"end":     day + "T16:00:00Z",
	}, &res)

	assert.Equal(t, "grouped", res.Mode)
	assert.Greater(t, res.TotalMatches, maxLogResults, "the window has far more lines than the raw cap")
	assert.False(t, res.Truncated)
	assert.Equal(t, res.TotalGroups, res.ReturnedGroups)

	// Hundreds of identical error lines collapse into a handful of groups
	// whose counts still add up to every matching line.
	require.NotEmpty(t, res.Groups)
	assert.Less(t, len(res.Groups), 5)
	sum := 0
	for _, g := range res.Groups {
		assert.Equal(t, "payments", g.Service)
		assert.Equal(t, "ERROR", g.Level)
		assert.LessOrEqual(t, g.FirstSeen, g.LastSeen)
		assert.LessOrEqual(t, len(g.SampleTraceIDs), maxSampleTraceIDs)
		sum += g.Count
	}
	assert.Equal(t, res.TotalMatches, sum)
}

// This is the reason grouping is the default. Over the incident window the
// 40 newest raw payments lines are all repeated errors (plus the rollback),
// and the deploy that caused the incident is cut off. Grouped, it survives
// as its own row, and the timeline shows errors starting right after it.
func TestQueryLogsGroupingKeepsRareDecisiveLines(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)
	args := map[string]string{
		"service": "payments",
		"start":   day + "T14:00:00Z",
		"end":     day + "T16:00:00Z",
	}
	isBadDeploy := func(msg string) bool { return strings.Contains(msg, "rolling out payments v2.14.0") }

	args["mode"] = "raw"
	var raw rawLogsResult
	mustCall(t, store, "query_logs", args, &raw)
	require.True(t, raw.Truncated)
	for _, e := range raw.Entries {
		assert.False(t, isBadDeploy(e.Message), "precondition: raw truncation hides the deploy line")
	}

	args["mode"] = "grouped"
	var grouped groupedLogsResult
	mustCall(t, store, "query_logs", args, &grouped)

	deployIdx, errorIdx := -1, -1
	for i, g := range grouped.Groups {
		if isBadDeploy(g.Message) {
			deployIdx = i
		}
		if g.Level == "ERROR" && errorIdx == -1 {
			errorIdx = i
		}
	}
	require.NotEqual(t, -1, deployIdx, "grouped output must contain the v2.14.0 deploy line")
	require.NotEqual(t, -1, errorIdx)
	assert.Less(t, deployIdx, errorIdx, "groups are a timeline: the deploy comes before the errors")
	assert.LessOrEqual(t, grouped.Groups[deployIdx].FirstSeen, grouped.Groups[errorIdx].FirstSeen)
}

func TestQueryLogsRawRespectsCapAndFilters(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)

	var res rawLogsResult
	mustCall(t, store, "query_logs", map[string]string{
		"service": "payments",
		"level":   "ERROR",
		"mode":    "raw",
		"start":   day + "T14:00:00Z",
		"end":     day + "T16:00:00Z",
	}, &res)

	assert.Equal(t, "raw", res.Mode)
	assert.True(t, res.Truncated)
	assert.Equal(t, maxLogResults, res.Returned)
	assert.Len(t, res.Entries, maxLogResults)
	assert.Greater(t, res.TotalMatches, res.Returned)
	for i, e := range res.Entries {
		assert.Equal(t, "payments", e.Service)
		assert.Equal(t, "ERROR", e.Level)
		if i > 0 {
			assert.GreaterOrEqual(t, res.Entries[i-1].TS, e.TS, "raw entries are most recent first")
		}
	}
}

// The workflow the tool description suggests: take a sample trace_id from a
// group, then pull that one request's lines across all services.
func TestQueryLogsFollowTraceAcrossServices(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)
	window := map[string]string{"start": day + "T14:00:00Z", "end": day + "T16:00:00Z"}

	var grouped groupedLogsResult
	mustCall(t, store, "query_logs", map[string]string{
		"service": "frontend", "level": "ERROR", "start": window["start"], "end": window["end"],
	}, &grouped)
	require.NotEmpty(t, grouped.Groups)
	require.NotEmpty(t, grouped.Groups[0].SampleTraceIDs)
	traceID := grouped.Groups[0].SampleTraceIDs[0]

	var raw rawLogsResult
	mustCall(t, store, "query_logs", map[string]string{
		"trace_id": traceID, "mode": "raw", "start": window["start"], "end": window["end"],
	}, &raw)

	services := map[string]bool{}
	for _, e := range raw.Entries {
		assert.Equal(t, traceID, e.TraceID)
		services[e.Service] = true
	}
	assert.True(t, services["frontend"] && services["checkout"] && services["payments"],
		"one failing request should be traceable frontend -> checkout -> payments, got %v", services)
}

func TestQueryLogsUnknownModeErrors(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)

	_, err := call(t, store, "query_logs", map[string]string{
		"mode": "summary", "start": day + "T14:00:00Z", "end": day + "T16:00:00Z",
	})
	assert.ErrorContains(t, err, `unknown mode "summary"`)
}

// groupLogs on synthetic input, to pin down ordering, sample and tie rules
// independently of the generated dataset.
func TestGroupLogs(t *testing.T) {
	entries := []data.LogEntry{
		// deliberately out of order
		{TS: "2026-01-01T10:05:00Z", Service: "b", Level: "ERROR", Message: "boom", TraceID: "t3"},
		{TS: "2026-01-01T10:00:00Z", Service: "a", Level: "INFO", Message: "deploy"},
		{TS: "2026-01-01T10:01:00Z", Service: "b", Level: "ERROR", Message: "boom", TraceID: "t1"},
		{TS: "2026-01-01T10:02:00Z", Service: "b", Level: "ERROR", Message: "boom", TraceID: "t1"},
		{TS: "2026-01-01T10:03:00Z", Service: "b", Level: "ERROR", Message: "boom", TraceID: "t2"},
		{TS: "2026-01-01T10:04:00Z", Service: "b", Level: "ERROR", Message: "boom", TraceID: "t4"},
		{TS: "2026-01-01T10:01:00Z", Service: "a", Level: "WARN", Message: "slow"},
		// same message, different service: a separate group
		{TS: "2026-01-01T10:06:00Z", Service: "c", Level: "ERROR", Message: "boom"},
	}

	groups := groupLogs(entries)

	require.Len(t, groups, 4)
	assert.Equal(t, LogGroup{
		Service: "a", Level: "INFO", Message: "deploy", Count: 1,
		FirstSeen: "2026-01-01T10:00:00Z", LastSeen: "2026-01-01T10:00:00Z",
	}, groups[0])
	// a/slow and b/boom both start at 10:01; ties break by service.
	assert.Equal(t, "slow", groups[1].Message)
	assert.Equal(t, LogGroup{
		Service: "b", Level: "ERROR", Message: "boom", Count: 5,
		FirstSeen: "2026-01-01T10:01:00Z", LastSeen: "2026-01-01T10:05:00Z",
		// earliest distinct trace_ids, capped at maxSampleTraceIDs
		SampleTraceIDs: []string{"t1", "t2", "t4"},
	}, groups[2])
	assert.Equal(t, "c", groups[3].Service)
}

func TestGetDeploysFindsIncidentDeploy(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)

	var deploys []data.Deploy
	mustCall(t, store, "get_deploys", map[string]string{
		"service": "payments",
		"start":   day + "T14:00:00Z",
		"end":     day + "T15:00:00Z",
	}, &deploys)

	require.Len(t, deploys, 1, "expected exactly 1 deploy in this window")
	assert.Equal(t, "v2.14.0", deploys[0].Version, "expected the incident-causing deploy")
}
