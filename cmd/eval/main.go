// Command eval runs the agent against the scenarios in evals/ and reports
// how often it finds the known root cause, and at what cost (roadmap stage
// 4). Each scenario's questions are asked -repeat times; every run is graded
// on its structured report and measured from its trace.
//
// Against a local model (free):
//
//	go run ./cmd/eval -base-url http://localhost:11434 -model qwen3-coder
//
// Against Anthropic (costs money: every question is a full investigation):
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./cmd/eval
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/richardknop/my-agent/internal/agent"
	"github.com/richardknop/my-agent/internal/claude"
	"github.com/richardknop/my-agent/internal/eval"
)

func main() {
	scenariosDir := flag.String("scenarios", "evals", "directory containing one subdirectory per scenario")
	filter := flag.String("run", "", "only run scenarios whose name contains this")
	repeat := flag.Int("repeat", 1, "how many times to ask each question")
	resultsDir := flag.String("results-dir", "evals/results", "directory to write the results JSON file into")
	traceDir := flag.String("trace-dir", "traces", "directory for the per-run traces (a subdirectory per eval)")
	model := flag.String("model", agent.DefaultConfig().Model, "model id (a Claude model, or a local model name such as qwen3-coder when using -base-url)")
	maxTurns := flag.Int("max-turns", agent.DefaultConfig().MaxTurns, "turn budget per run")
	maxTokens := flag.Int("max-tokens", agent.DefaultConfig().MaxTokens, "max tokens per model response")
	baseURL := flag.String("base-url", os.Getenv("ANTHROPIC_BASE_URL"), "Messages API base URL; empty = Anthropic, e.g. http://localhost:11434 for Ollama (defaults to $ANTHROPIC_BASE_URL)")
	timeout := flag.Duration("timeout", 10*time.Minute, "time limit for each run")
	verbosity := flag.Int("v", 0, "agent step log on stderr: 0 = only one line per run, 1 = each loop step, 2 = full payloads")
	flag.Parse()

	if *repeat < 1 {
		log.Fatal("-repeat must be at least 1")
	}
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" && claude.IsAnthropicHosted(*baseURL) {
		log.Fatal("ANTHROPIC_API_KEY is not set. Export it, or run against a local model with " +
			"-base-url http://localhost:11434 -model qwen3-coder")
	}

	scenarios, err := eval.LoadScenarios(*scenariosDir, *filter)
	if err != nil {
		log.Fatal(err)
	}
	total := 0
	for _, sc := range scenarios {
		total += len(sc.Questions) * *repeat
	}

	started := time.Now().UTC()
	stamp := started.Format("20060102T150405Z")
	runTraceDir := filepath.Join(*traceDir, "eval-"+stamp)
	if err := os.MkdirAll(runTraceDir, 0o755); err != nil {
		log.Fatalf("creating trace dir: %v", err)
	}

	cfg := agent.DefaultConfig()
	cfg.Model = *model
	cfg.MaxTurns = *maxTurns
	cfg.MaxTokens = *maxTokens
	cfg.Verbosity = *verbosity
	cfg.LogWriter = os.Stderr
	runner := &eval.Runner{
		LLM:      claude.NewClient(apiKey, *baseURL),
		Config:   cfg,
		TraceDir: runTraceDir,
		Timeout:  *timeout,
	}

	fmt.Fprintf(os.Stderr, "eval: %d scenario(s), %d run(s) against %s (model %s)\n\n",
		len(scenarios), total, claude.ResolveURL(*baseURL), *model)

	// Ctrl-C stops after the current run and still writes what finished:
	// a half-done eval of a slow local model is worth keeping.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	results := eval.Results{
		StartedAt: started,
		Commit:    gitCommit(),
		Model:     *model,
		BaseURL:   claude.ResolveURL(*baseURL),
		MaxTurns:  *maxTurns,
		MaxTokens: *maxTokens,
		Repeat:    *repeat,
	}
	done := 0
	for _, sc := range scenarios {
		runs, err := runner.RunScenario(ctx, sc, *repeat, func(r eval.RunResult) {
			done++
			fmt.Fprintf(os.Stderr, "[%d/%d] %s\n", done, total, describe(r))
		})
		results.Runs = append(results.Runs, runs...)
		if len(runs) > 0 {
			results.Summaries = append(results.Summaries, eval.Summarize(sc.Name, runs))
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "eval stopped: %v\n", err)
			break
		}
	}
	if len(results.Summaries) > 1 {
		results.Summaries = append(results.Summaries, eval.Summarize("all", results.Runs))
	}

	fmt.Println()
	if err := eval.WriteTable(os.Stdout, results.Summaries); err != nil {
		log.Fatal(err)
	}

	path := filepath.Join(*resultsDir, fmt.Sprintf("%s-%s.json", stamp, sanitize(*model)))
	if err := results.Write(path); err != nil {
		log.Fatalf("writing results: %v", err)
	}
	fmt.Fprintf(os.Stderr, "\nresults written to %s, traces in %s\n", path, runTraceDir)
}

// describe is the one-line progress report for a run.
func describe(r eval.RunResult) string {
	verdict := "FAIL"
	if r.Grade.Pass {
		verdict = "PASS"
	}
	head := fmt.Sprintf("%s q%d r%d  %s", r.Scenario, r.Question, r.Repeat, verdict)
	stats := fmt.Sprintf("turns=%d rejected=%d %.0fs", r.Stats.Turns, r.Stats.RejectedReports,
		float64(r.Stats.DurationMS)/1000)
	if r.Report == nil {
		return fmt.Sprintf("%s  error: %s  %s", head, r.Error, stats)
	}
	return fmt.Sprintf("%s  service=%q change=%q affected=%.0f%%  %s", head,
		r.Report.RootCauseService, r.Report.TriggeringChange, r.Grade.AffectedRecall*100, stats)
}

// gitCommit identifies the code under evaluation, so results from different
// commits can be compared. Uncommitted changes are flagged: a result is only
// reproducible from a clean tree.
func gitCommit() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	commit := strings.TrimSpace(string(out))
	if status, err := exec.Command("git", "status", "--porcelain", "--untracked-files=no").Output(); err == nil && len(status) > 0 {
		commit += "+dirty"
	}
	return commit
}

// sanitize makes a model id safe for a file name ("qwen3-coder:30b" ->
// "qwen3-coder_30b").
func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '.':
			return r
		}
		return '_'
	}, s)
}
