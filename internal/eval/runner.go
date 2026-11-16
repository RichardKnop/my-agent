package eval

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"github.com/richardknop/my-agent/internal/agent"
	"github.com/richardknop/my-agent/internal/data"
	"github.com/richardknop/my-agent/internal/tools"
	"github.com/richardknop/my-agent/internal/trace"
)

// Runner runs the real agent loop against scenarios. It takes the model as
// an agent.LLM, so tests drive it with a scripted fake and cmd/eval with the
// real client.
type Runner struct {
	LLM agent.LLM
	// Config is the agent configuration for every run; Now is overridden
	// per scenario.
	Config   agent.Config
	TraceDir string
	// Timeout bounds each run; zero means no limit.
	Timeout time.Duration
}

// RunResult is one agent run on one question, graded.
type RunResult struct {
	Scenario string `json:"scenario"`
	// Question is the index into the scenario's questions, so results from
	// different commits line up even if the wording is later edited.
	Question     int           `json:"question"`
	QuestionText string        `json:"question_text"`
	Repeat       int           `json:"repeat"`
	Error        string        `json:"error,omitempty"`
	Report       *agent.Report `json:"report,omitempty"`
	Grade        Grade         `json:"grade"`
	Stats        RunStats      `json:"stats"`
	TracePath    string        `json:"trace_path"`
}

// ErrModelUnavailable stops an eval when a model API call fails. The client
// has already retried, so the endpoint is down or rejecting requests, and
// every remaining run would fail the same way. Carrying on would record an
// outage as agent failures.
var ErrModelUnavailable = errors.New("model API call failed; stopping the eval")

// RunScenario runs every question of a scenario repeat times, calling
// onResult after each run (for progress output), and returns all results.
// It stops early, returning what finished, on cancellation or
// ErrModelUnavailable.
func (r *Runner) RunScenario(ctx context.Context, sc Scenario, repeat int, onResult func(RunResult)) ([]RunResult, error) {
	store, err := data.Load(sc.DataPath())
	if err != nil {
		return nil, fmt.Errorf("scenario %s: loading data: %w", sc.Name, err)
	}
	now, err := sc.NowTime()
	if err != nil {
		return nil, fmt.Errorf("scenario %s: %w", sc.Name, err)
	}

	var results []RunResult
	for q, question := range sc.Questions {
		for rep := 1; rep <= repeat; rep++ {
			res := r.runOne(ctx, sc, store, now, q, question, rep)
			results = append(results, res)
			if onResult != nil {
				onResult(res)
			}
			if ctx.Err() != nil {
				return results, ctx.Err()
			}
			if res.Stats.ModelError {
				return results, fmt.Errorf("%w: %s", ErrModelUnavailable, res.Error)
			}
		}
	}
	return results, nil
}

func (r *Runner) runOne(ctx context.Context, sc Scenario, store *data.Store, now time.Time, q int, question string, rep int) RunResult {
	res := RunResult{
		Scenario: sc.Name, Question: q, QuestionText: question, Repeat: rep,
		TracePath: filepath.Join(r.TraceDir, fmt.Sprintf("%s-q%d-r%d.jsonl", sc.Name, q, rep)),
	}

	tracer, err := trace.New(res.TracePath)
	if err != nil {
		res.Error = fmt.Sprintf("creating trace: %v", err)
		return res
	}

	cfg := r.Config
	cfg.Now = now
	a := agent.New(r.LLM, tools.NewRegistry(store), tracer, cfg)

	runCtx := ctx
	if r.Timeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}
	report, runErr := a.Investigate(runCtx, question)
	if err := tracer.Close(); err != nil && runErr == nil {
		runErr = fmt.Errorf("closing trace: %w", err)
	}

	res.Report = report
	res.Grade = GradeReport(sc.Expect, store.Deploys, report)
	if runErr != nil {
		res.Error = runErr.Error()
	}
	stats, err := ReadTrace(res.TracePath)
	if err != nil {
		res.Error = joinErr(res.Error, fmt.Sprintf("reading trace: %v", err))
	}
	res.Stats = stats
	return res
}

func joinErr(a, b string) string {
	if a == "" {
		return b
	}
	return a + "; " + b
}
