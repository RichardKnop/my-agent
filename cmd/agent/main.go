// Command agent is the CLI entrypoint: point it at a question and it runs
// the incident-investigation agent loop against the mock dataset in data/.
//
// Usage:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//	go run ./cmd/agent "why did checkout error rate spike yesterday afternoon?"
//
// Or against a local model served by Ollama (no API key, no cost):
//
//	go run ./cmd/agent -base-url http://localhost:11434 -model qwen3-coder "..."
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/richardknop/my-agent/internal/agent"
	"github.com/richardknop/my-agent/internal/claude"
	"github.com/richardknop/my-agent/internal/data"
	"github.com/richardknop/my-agent/internal/tools"
	"github.com/richardknop/my-agent/internal/trace"
)

func main() {
	dataDir := flag.String("data-dir", "data", "directory containing services.json, deploys.json, metrics.json, logs.json")
	traceDir := flag.String("trace-dir", "traces", "directory to write the per-run JSONL trace file into")
	model := flag.String("model", agent.DefaultConfig().Model, "model id (a Claude model, or a local model name such as qwen3-coder when using -base-url)")
	maxTurns := flag.Int("max-turns", agent.DefaultConfig().MaxTurns, "max agent loop turns before giving up")
	baseURL := flag.String("base-url", os.Getenv("ANTHROPIC_BASE_URL"), "Messages API base URL; empty = Anthropic, e.g. http://localhost:11434 for Ollama (defaults to $ANTHROPIC_BASE_URL)")
	timeout := flag.Duration("timeout", 3*time.Minute, "overall time limit for one investigation (local models may need more, e.g. 10m)")
	verbosity := flag.Int("v", 1, "step log on stderr: 0 = silent, 1 = each loop step, 2 = each step plus full payloads (system prompt, tool definitions, tool results)")
	nowFlag := flag.String("now", "", "RFC3339 time the agent treats as \"now\" (default: real current time); use to replay an older dataset, e.g. 2026-08-24T09:00:00Z")
	flag.Parse()

	var now time.Time
	if *nowFlag != "" {
		var err error
		if now, err = time.Parse(time.RFC3339, *nowFlag); err != nil {
			log.Fatalf("invalid -now %q: %v", *nowFlag, err)
		}
	}

	question := strings.Join(flag.Args(), " ")
	if question == "" {
		question = "Why did checkout's error rate spike yesterday afternoon, and what fixed it?"
		fmt.Fprintf(os.Stderr, "no question given, using default: %q\n\n", question)
	}

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" && claude.IsAnthropicHosted(*baseURL) {
		log.Fatal("ANTHROPIC_API_KEY is not set. Get a key from console.anthropic.com and export it, " +
			"or run against a local model with e.g. -base-url http://localhost:11434 -model qwen3-coder")
	}

	store, err := data.Load(*dataDir)
	if err != nil {
		log.Fatalf("loading dataset from %s: %v (did you run `python3 scripts/gen_data.py` first?)", *dataDir, err)
	}

	if err := os.MkdirAll(*traceDir, 0o755); err != nil {
		log.Fatalf("creating trace dir: %v", err)
	}
	tracePath := filepath.Join(*traceDir, fmt.Sprintf("run-%s.jsonl", time.Now().UTC().Format("20060102T150405Z")))
	tracer, err := trace.New(tracePath)
	if err != nil {
		log.Fatalf("creating trace file: %v", err)
	}
	defer tracer.Close()

	client := claude.NewClient(apiKey, *baseURL)
	if *verbosity > 0 {
		fmt.Fprintf(os.Stderr, "using %s (model %s)\n\n", claude.ResolveURL(*baseURL), *model)
	}
	registry := tools.NewRegistry(store)

	cfg := agent.DefaultConfig()
	cfg.Model = *model
	cfg.MaxTurns = *maxTurns
	cfg.Now = now
	cfg.Verbosity = *verbosity
	cfg.LogWriter = os.Stderr

	a := agent.New(client, registry, tracer, cfg)

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	answer, err := a.Investigate(ctx, question)
	if err != nil {
		log.Fatalf("agent run failed: %v\n(trace written to %s)", err, tracePath)
	}

	fmt.Println(answer)
	fmt.Fprintf(os.Stderr, "\n--- trace written to %s ---\n", tracePath)
}
