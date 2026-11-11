package tools

import (
	"encoding/json"
	"path/filepath"
	"runtime"
	"testing"

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
	if err != nil {
		t.Fatalf("loading dataset (did you run `python3 scripts/gen_data.py`?): %v", err)
	}
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

func TestListServices(t *testing.T) {
	store := loadTestStore(t)
	reg := NewRegistry()

	out, err := reg.Call("list_services", store, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var services []data.Service
	if err := json.Unmarshal([]byte(out), &services); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(services) == 0 {
		t.Fatal("expected at least one service")
	}

	names := map[string]bool{}
	for _, s := range services {
		names[s.Name] = true
	}
	for _, want := range []string{"frontend", "checkout", "payments", "payments-db"} {
		if !names[want] {
			t.Errorf("expected service %q in catalog", want)
		}
	}
}

func TestQueryMetricsFindsIncidentSpike(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)
	reg := NewRegistry()

	input, _ := json.Marshal(map[string]string{
		"service": "payments",
		"metric":  "error_rate",
		"start":   day + "T14:00:00Z",
		"end":     day + "T15:00:00Z",
	})

	out, err := reg.Call("query_metrics", store, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var result struct {
		Summary struct {
			Max float64 `json:"max"`
		} `json:"summary"`
	}
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Baseline error_rate for payments is ~0.2%; the incident should push
	// the max well past that within this window.
	if result.Summary.Max < 5.0 {
		t.Errorf("expected a clear error_rate spike for payments in the incident window, got max=%.2f", result.Summary.Max)
	}
}

func TestQueryMetricsUnknownServiceErrors(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)
	reg := NewRegistry()

	input, _ := json.Marshal(map[string]string{
		"service": "does-not-exist",
		"metric":  "error_rate",
		"start":   day + "T14:00:00Z",
		"end":     day + "T15:00:00Z",
	})

	if _, err := reg.Call("query_metrics", store, input); err == nil {
		t.Fatal("expected an error for an unknown service")
	}
}

func TestQueryLogsRespectsCapAndFilters(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)
	reg := NewRegistry()

	input, _ := json.Marshal(map[string]string{
		"service": "payments",
		"level":   "ERROR",
		"start":   day + "T14:00:00Z",
		"end":     day + "T16:00:00Z",
	})

	out, err := reg.Call("query_logs", store, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var result struct {
		TotalMatches int             `json:"total_matches"`
		Returned     int             `json:"returned"`
		Truncated    bool            `json:"truncated"`
		Entries      []data.LogEntry `json:"entries"`
	}
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if result.Returned > maxLogResults {
		t.Errorf("returned %d entries, expected at most %d", result.Returned, maxLogResults)
	}
	if result.TotalMatches < result.Returned {
		t.Errorf("total_matches (%d) should be >= returned (%d)", result.TotalMatches, result.Returned)
	}
	for _, e := range result.Entries {
		if e.Service != "payments" {
			t.Errorf("expected only payments logs, got service %q", e.Service)
		}
		if e.Level != "ERROR" {
			t.Errorf("expected only ERROR logs, got level %q", e.Level)
		}
	}
}

func TestGetDeploysFindsIncidentDeploy(t *testing.T) {
	store := loadTestStore(t)
	day := incidentDay(t, store)
	reg := NewRegistry()

	input, _ := json.Marshal(map[string]string{
		"service": "payments",
		"start":   day + "T14:00:00Z",
		"end":     day + "T15:00:00Z",
	})

	out, err := reg.Call("get_deploys", store, input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var deploys []data.Deploy
	if err := json.Unmarshal([]byte(out), &deploys); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(deploys) != 1 {
		t.Fatalf("expected exactly 1 deploy in this window, got %d", len(deploys))
	}
	if deploys[0].Version != "v2.14.0" {
		t.Errorf("expected the incident-causing deploy v2.14.0, got %q", deploys[0].Version)
	}
}
