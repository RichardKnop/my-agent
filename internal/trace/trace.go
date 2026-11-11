// Package trace is the smallest possible version of "agent observability":
// one JSONL file per run with a timestamped event per LLM call and per
// tool call, including token usage and latency. Real systems would ship
// this to ClickHouse/OTel instead of a local file — the shape of the data
// (spans with duration + attributes) doesn't change, just the sink.
package trace

import (
	"encoding/json"
	"os"
	"time"
)

type Tracer struct {
	file    *os.File
	enc     *json.Encoder
	started time.Time
}

func New(path string) (*Tracer, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &Tracer{file: f, enc: json.NewEncoder(f), started: time.Now()}, nil
}

// Event appends one JSON line: {"ts", "elapsed_ms", "event", ...fields}.
func (t *Tracer) Event(event string, fields map[string]any) {
	if t == nil {
		return
	}
	rec := map[string]any{
		"ts":         time.Now().UTC().Format(time.RFC3339Nano),
		"elapsed_ms": time.Since(t.started).Milliseconds(),
		"event":      event,
	}
	for k, v := range fields {
		rec[k] = v
	}
	_ = t.enc.Encode(rec)
}

func (t *Tracer) Close() error {
	if t == nil {
		return nil
	}
	return t.file.Close()
}
