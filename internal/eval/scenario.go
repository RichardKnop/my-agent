// Package eval is roadmap stage 4: run the agent against incident scenarios
// with known root causes and measure how often it gets them right, and at
// what cost.
//
// An eval is like an integration test with one big difference: the system
// under test is non-deterministic, so a single pass or fail means little. The
// unit of measurement is a rate over many runs ("root cause found in 5 of 6
// phrasings, 7.2 turns on average"), and a regression is a rate dropping
// after a prompt, tool or model change.
//
// Grading is deterministic on purpose. The agent answers with a structured
// agent.Report, so fields like root_cause_service are compared with the
// expected values directly: no phrase matching (which breaks when a model
// words things differently) and no grader model (which costs money and has
// its own variance). Process measurements (turns, tokens, tool errors,
// rejected reports) come from the run's JSONL trace.
package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Scenario is one incident with a known answer, loaded from
// <root>/<name>/scenario.json. The expected answer lives only here: nothing
// in it may ever reach the agent's prompts, or the eval measures nothing.
type Scenario struct {
	Name string `json:"-"` // the scenario's directory name
	Dir  string `json:"-"`

	Description string `json:"description"`
	// DataDir holds the dataset (services.json, metrics.json, ...), relative
	// to the scenario directory. Empty means "data" inside it.
	DataDir string `json:"data_dir"`
	// Now is the RFC3339 time the agent is told it is, so that relative
	// questions ("yesterday afternoon") point at the incident.
	Now string `json:"now"`
	// Questions are different phrasings of the same investigation. They are
	// the main source of variety: the same question asked twice of a local
	// model often produces nearly the same run.
	Questions []string `json:"questions"`
	Expect    Expect   `json:"expect"`
}

// Expect is the known answer a report is graded against.
type Expect struct {
	// RootCauseService must equal the report's root_cause_service (ignoring
	// case). Empty means the scenario has no incident.
	RootCauseService string `json:"root_cause_service"`
	// TriggeringChange must appear somewhere in the report's
	// triggering_change (ignoring case), because models phrase it freely:
	// "v2.14.0", "payments v2.14.0", "v2.14.0 deploy to payments". Empty
	// means no change triggered the incident, so the report must leave it
	// empty too.
	TriggeringChange string `json:"triggering_change"`
	// AffectedServices are the services that showed impact. A report is
	// scored on how many of them it names, not pass/fail: missing a
	// downstream service is a weaker answer, not a wrong one.
	AffectedServices []string `json:"affected_services"`
	// UnaffectedServices are healthy controls. Naming one as affected is
	// recorded as a localization mistake.
	UnaffectedServices []string `json:"unaffected_services"`
}

// LoadScenarios loads every scenario directory under root whose name
// contains filter (all of them if filter is empty), sorted by name.
func LoadScenarios(root, filter string) ([]Scenario, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("reading scenarios: %w", err)
	}
	var out []Scenario
	for _, e := range entries {
		if !e.IsDir() || !strings.Contains(e.Name(), filter) {
			continue
		}
		if _, err := os.Stat(filepath.Join(root, e.Name(), "scenario.json")); err != nil {
			continue // not a scenario, e.g. the results directory
		}
		sc, err := LoadScenario(filepath.Join(root, e.Name()))
		if err != nil {
			return nil, err
		}
		out = append(out, sc)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no scenarios under %s matching %q", root, filter)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// LoadScenario loads and checks one scenario directory.
func LoadScenario(dir string) (Scenario, error) {
	sc := Scenario{Name: filepath.Base(dir), Dir: dir}
	b, err := os.ReadFile(filepath.Join(dir, "scenario.json"))
	if err != nil {
		return sc, fmt.Errorf("scenario %s: %w", sc.Name, err)
	}
	// Unknown fields are an error: a misspelled expectation such as
	// "root_cause" would otherwise be ignored, and every run would be graded
	// against an empty expectation.
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&sc); err != nil {
		return sc, fmt.Errorf("scenario %s: invalid scenario.json: %w", sc.Name, err)
	}
	if len(sc.Questions) == 0 {
		return sc, fmt.Errorf("scenario %s: no questions", sc.Name)
	}
	if _, err := sc.NowTime(); err != nil {
		return sc, fmt.Errorf("scenario %s: %w", sc.Name, err)
	}
	if _, err := os.Stat(sc.DataPath()); err != nil {
		return sc, fmt.Errorf("scenario %s: data directory: %w", sc.Name, err)
	}
	return sc, nil
}

// DataPath is the dataset directory, resolved against the scenario directory.
func (s Scenario) DataPath() string {
	if s.DataDir == "" {
		return filepath.Join(s.Dir, "data")
	}
	return filepath.Join(s.Dir, s.DataDir)
}

// NowTime parses Now.
func (s Scenario) NowTime() (time.Time, error) {
	t, err := time.Parse(time.RFC3339, s.Now)
	if err != nil {
		return time.Time{}, fmt.Errorf("invalid now %q: must be an RFC3339 timestamp", s.Now)
	}
	return t, nil
}
