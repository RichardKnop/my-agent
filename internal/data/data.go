// Package data loads the mock observability dataset (services, deploys,
// metrics, logs) that the agent's tools read from. In stage 6 of the
// roadmap (see README) this package's Store is swapped for one backed by
// real ClickHouse queries — the tool layer above it doesn't need to change.
package data

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Service struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	DependsOn   []string `json:"depends_on"`
}

type Deploy struct {
	Service   string `json:"service"`
	Version   string `json:"version"`
	Timestamp string `json:"timestamp"`
	Author    string `json:"author"`
	Notes     string `json:"notes"`
}

type MetricPoint struct {
	TS    string  `json:"ts"`
	Value float64 `json:"value"`
}

type LogEntry struct {
	TS      string `json:"ts"`
	Service string `json:"service"`
	Level   string `json:"level"`
	Message string `json:"message"`
	TraceID string `json:"trace_id,omitempty"`
}

// Store holds the whole mock dataset in memory. Metrics is keyed
// [service][metric] -> time-ordered points.
type Store struct {
	Services []Service
	Deploys  []Deploy
	Metrics  map[string]map[string][]MetricPoint
	Logs     []LogEntry
}

// Load reads services.json, deploys.json, metrics.json and logs.json from
// dataDir. The generator that produced them lives at scripts/gen_data.py.
func Load(dataDir string) (*Store, error) {
	s := &Store{}

	if err := readJSON(filepath.Join(dataDir, "services.json"), &s.Services); err != nil {
		return nil, fmt.Errorf("load services: %w", err)
	}
	if err := readJSON(filepath.Join(dataDir, "deploys.json"), &s.Deploys); err != nil {
		return nil, fmt.Errorf("load deploys: %w", err)
	}
	if err := readJSON(filepath.Join(dataDir, "metrics.json"), &s.Metrics); err != nil {
		return nil, fmt.Errorf("load metrics: %w", err)
	}
	if err := readJSON(filepath.Join(dataDir, "logs.json"), &s.Logs); err != nil {
		return nil, fmt.Errorf("load logs: %w", err)
	}

	return s, nil
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}
