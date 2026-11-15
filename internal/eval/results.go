package eval

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"
	"time"
)

// Summary aggregates runs into the rates an eval is really about.
type Summary struct {
	Scenario string `json:"scenario"` // "all" for the overall row
	Runs     int    `json:"runs"`
	Passed   int    `json:"passed"`
	// RootCauseService and TriggeringChange count runs that got each field
	// right, to show which part of the answer fails.
	RootCauseService int     `json:"root_cause_service"`
	TriggeringChange int     `json:"triggering_change"`
	AffectedRecall   float64 `json:"affected_recall"` // mean over runs
	WronglyAffected  int     `json:"wrongly_affected"`
	// Errors counts runs that ended without a report, including any
	// model API failure that stopped the eval.
	Errors          int     `json:"errors"`
	BudgetExhausted int     `json:"budget_exhausted"`
	RejectedReports int     `json:"rejected_reports"`
	ToolErrors      int     `json:"tool_errors"`
	AvgTurns        float64 `json:"avg_turns"`
	AvgInputTokens  float64 `json:"avg_input_tokens"`
	AvgCacheTokens  float64 `json:"avg_cache_read_input_tokens"`
	AvgOutputTokens float64 `json:"avg_output_tokens"`
	AvgDurationMS   float64 `json:"avg_duration_ms"`
}

// Summarize aggregates a set of runs under one name.
func Summarize(name string, runs []RunResult) Summary {
	s := Summary{Scenario: name, Runs: len(runs)}
	if len(runs) == 0 {
		return s
	}
	for _, r := range runs {
		if r.Grade.Pass {
			s.Passed++
		}
		if r.Grade.RootCauseService {
			s.RootCauseService++
		}
		if r.Grade.TriggeringChange {
			s.TriggeringChange++
		}
		if len(r.Grade.WronglyAffected) > 0 {
			s.WronglyAffected++
		}
		if r.Report == nil {
			s.Errors++
		}
		if r.Stats.BudgetExhausted {
			s.BudgetExhausted++
		}
		s.AffectedRecall += r.Grade.AffectedRecall
		s.RejectedReports += r.Stats.RejectedReports
		s.ToolErrors += r.Stats.ToolErrors
		s.AvgTurns += float64(r.Stats.Turns)
		s.AvgInputTokens += float64(r.Stats.InputTokens)
		s.AvgCacheTokens += float64(r.Stats.CacheReadTokens)
		s.AvgOutputTokens += float64(r.Stats.OutputTokens)
		s.AvgDurationMS += float64(r.Stats.DurationMS)
	}
	n := float64(len(runs))
	s.AffectedRecall /= n
	s.AvgTurns /= n
	s.AvgInputTokens /= n
	s.AvgCacheTokens /= n
	s.AvgOutputTokens /= n
	s.AvgDurationMS /= n
	return s
}

// Results is everything one eval invocation produced. Written as JSON, it's
// the record to compare across commits, prompts and models.
type Results struct {
	StartedAt time.Time `json:"started_at"`
	// Commit is the git commit the agent was built from, "+dirty" if the
	// working tree had uncommitted changes.
	Commit    string      `json:"commit"`
	Model     string      `json:"model"`
	BaseURL   string      `json:"base_url"`
	MaxTurns  int         `json:"max_turns"`
	MaxTokens int         `json:"max_tokens"`
	Repeat    int         `json:"repeat"`
	Summaries []Summary   `json:"summaries"`
	Runs      []RunResult `json:"runs"`
}

// Write saves the results as indented JSON, creating the directory if needed.
func (r *Results) Write(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// WriteTable prints summaries as an aligned table.
func WriteTable(w io.Writer, sums []Summary) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "scenario\tpass\troot cause\tchange\taffected\terrors\tbudget out\trejected\ttool errs\tturns\ttokens in/cached/out\ttime\t")
	for _, s := range sums {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%.0f%%\t%d\t%d\t%d\t%d\t%.1f\t%s/%s/%s\t%.0fs\t\n",
			s.Scenario, frac(s.Passed, s.Runs), frac(s.RootCauseService, s.Runs), frac(s.TriggeringChange, s.Runs),
			s.AffectedRecall*100, s.Errors, s.BudgetExhausted, s.RejectedReports, s.ToolErrors,
			s.AvgTurns, kilo(s.AvgInputTokens), kilo(s.AvgCacheTokens), kilo(s.AvgOutputTokens), s.AvgDurationMS/1000)
	}
	return tw.Flush()
}

func frac(n, of int) string { return fmt.Sprintf("%d/%d", n, of) }

func kilo(v float64) string {
	if v < 1000 {
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.1fK", v/1000)
}
