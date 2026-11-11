// Package tools implements the small set of read-only tools the agent can
// call: list_services, query_metrics, query_logs and get_deploys. Each tool
// has a Go implementation plus a JSON-schema Definition that gets sent to
// Claude in the `tools` field of the Messages API request — this is the
// same shape an MCP server would expose (see roadmap stage 3 in README).
package tools

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/richardknop/my-agent/internal/data"
)

// Definition mirrors Anthropic's tool schema: name, a description the model
// uses to decide *when* to call the tool, and a JSON Schema for its input.
type Definition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// Handler executes one tool call against the dataset and returns the
// result already JSON-encoded, ready to go back to Claude as a tool_result.
type Handler func(store *data.Store, input json.RawMessage) (string, error)

// Registry pairs each tool's schema with its handler so the agent loop can
// look either up by name without a big switch statement.
type Registry struct {
	defs     []Definition
	handlers map[string]Handler
}

func NewRegistry() *Registry {
	r := &Registry{handlers: map[string]Handler{}}
	r.register(listServicesDef, listServices)
	r.register(queryMetricsDef, queryMetrics)
	r.register(queryLogsDef, queryLogs)
	r.register(getDeploysDef, getDeploys)
	return r
}

func (r *Registry) register(def Definition, h Handler) {
	r.defs = append(r.defs, def)
	r.handlers[def.Name] = h
}

func (r *Registry) Definitions() []Definition { return r.defs }

func (r *Registry) Call(name string, store *data.Store, input json.RawMessage) (string, error) {
	h, ok := r.handlers[name]
	if !ok {
		return "", fmt.Errorf("unknown tool %q", name)
	}
	return h(store, input)
}

// ---- list_services ----

var listServicesDef = Definition{
	Name: "list_services",
	Description: "List every service in the system catalog, with a one-line description and its " +
		"direct downstream dependencies (services it calls). Call this first to orient yourself " +
		"before querying metrics or logs for a specific service.",
	InputSchema: map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	},
}

func listServices(store *data.Store, _ json.RawMessage) (string, error) {
	return encode(store.Services)
}

// ---- query_metrics ----

var queryMetricsDef = Definition{
	Name: "query_metrics",
	Description: "Return a time series for one metric on one service, restricted to a time window. " +
		"Use this to check whether a service's error rate or latency actually moved, and when.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"service": map[string]any{
				"type":        "string",
				"description": "Service name, e.g. \"payments\". Call list_services to see valid names.",
			},
			"metric": map[string]any{
				"type": "string",
				"enum": []string{"error_rate", "p99_latency_ms"},
			},
			"start": map[string]any{
				"type":        "string",
				"description": "RFC3339 start timestamp, inclusive, e.g. \"2026-01-31T14:00:00Z\".",
			},
			"end": map[string]any{
				"type":        "string",
				"description": "RFC3339 end timestamp, exclusive.",
			},
		},
		"required": []string{"service", "metric", "start", "end"},
	},
}

type metricsArgs struct {
	Service string `json:"service"`
	Metric  string `json:"metric"`
	Start   string `json:"start"`
	End     string `json:"end"`
}

func queryMetrics(store *data.Store, input json.RawMessage) (string, error) {
	var a metricsArgs
	if err := json.Unmarshal(input, &a); err != nil {
		return "", fmt.Errorf("bad input: %w", err)
	}
	svc, ok := store.Metrics[a.Service]
	if !ok {
		return "", fmt.Errorf("unknown service %q", a.Service)
	}
	series, ok := svc[a.Metric]
	if !ok {
		return "", fmt.Errorf("unknown metric %q for service %q", a.Metric, a.Service)
	}

	var out []data.MetricPoint
	for _, p := range series {
		if inRange(p.TS, a.Start, a.End) {
			out = append(out, p)
		}
	}

	summary := summarize(out)
	return encode(map[string]any{
		"service": a.Service,
		"metric":  a.Metric,
		"points":  out,
		"summary": summary,
	})
}

func summarize(points []data.MetricPoint) map[string]any {
	if len(points) == 0 {
		return map[string]any{"count": 0}
	}
	min, max, sum := points[0].Value, points[0].Value, 0.0
	for _, p := range points {
		if p.Value < min {
			min = p.Value
		}
		if p.Value > max {
			max = p.Value
		}
		sum += p.Value
	}
	return map[string]any{
		"count": len(points),
		"min":   round2(min),
		"max":   round2(max),
		"avg":   round2(sum / float64(len(points))),
	}
}

func round2(f float64) float64 {
	return float64(int(f*100+0.5)) / 100
}

// ---- query_logs ----

const maxLogResults = 40

var queryLogsDef = Definition{
	Name: "query_logs",
	Description: fmt.Sprintf(
		"Search log lines within a time window, optionally filtered by service, level and a "+
			"substring match on the message. Results are capped at %d lines (most recent first) — "+
			"narrow the time window or add a \"contains\" filter if you need something more specific. "+
			"Use the shared trace_id on returned entries to follow one request across services.",
		maxLogResults),
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"service": map[string]any{
				"type":        "string",
				"description": "Optional service name filter.",
			},
			"level": map[string]any{
				"type": "string",
				"enum": []string{"INFO", "WARN", "ERROR"},
			},
			"contains": map[string]any{
				"type":        "string",
				"description": "Optional case-insensitive substring the log message must contain.",
			},
			"start": map[string]any{
				"type": "string",
			},
			"end": map[string]any{
				"type": "string",
			},
		},
		"required": []string{"start", "end"},
	},
}

type logsArgs struct {
	Service  string `json:"service"`
	Level    string `json:"level"`
	Contains string `json:"contains"`
	Start    string `json:"start"`
	End      string `json:"end"`
}

func queryLogs(store *data.Store, input json.RawMessage) (string, error) {
	var a logsArgs
	if err := json.Unmarshal(input, &a); err != nil {
		return "", fmt.Errorf("bad input: %w", err)
	}

	var matched []data.LogEntry
	for _, l := range store.Logs {
		if !inRange(l.TS, a.Start, a.End) {
			continue
		}
		if a.Service != "" && l.Service != a.Service {
			continue
		}
		if a.Level != "" && l.Level != a.Level {
			continue
		}
		if a.Contains != "" && !strings.Contains(strings.ToLower(l.Message), strings.ToLower(a.Contains)) {
			continue
		}
		matched = append(matched, l)
	}

	sort.Slice(matched, func(i, j int) bool { return matched[i].TS > matched[j].TS })

	total := len(matched)
	truncated := false
	if len(matched) > maxLogResults {
		matched = matched[:maxLogResults]
		truncated = true
	}

	return encode(map[string]any{
		"total_matches": total,
		"returned":      len(matched),
		"truncated":     truncated,
		"entries":       matched,
	})
}

// ---- get_deploys ----

var getDeploysDef = Definition{
	Name: "get_deploys",
	Description: "List deploy events (service, version, timestamp, author, notes) within a time " +
		"window, optionally filtered to one service. Deploys are one of the most common root causes " +
		"of incidents — always check what shipped shortly before a symptom started.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"service": map[string]any{
				"type":        "string",
				"description": "Optional service name filter.",
			},
			"start": map[string]any{"type": "string"},
			"end":   map[string]any{"type": "string"},
		},
		"required": []string{"start", "end"},
	},
}

type deploysArgs struct {
	Service string `json:"service"`
	Start   string `json:"start"`
	End     string `json:"end"`
}

func getDeploys(store *data.Store, input json.RawMessage) (string, error) {
	var a deploysArgs
	if err := json.Unmarshal(input, &a); err != nil {
		return "", fmt.Errorf("bad input: %w", err)
	}

	var out []data.Deploy
	for _, d := range store.Deploys {
		if !inRange(d.Timestamp, a.Start, a.End) {
			continue
		}
		if a.Service != "" && d.Service != a.Service {
			continue
		}
		out = append(out, d)
	}

	return encode(out)
}

// ---- shared helpers ----

// inRange does plain string comparison, which works because every
// timestamp in the dataset is RFC3339 in UTC ("2026-01-31T14:31:00Z") —
// that format sorts lexicographically in time order.
func inRange(ts, start, end string) bool {
	if start != "" && ts < start {
		return false
	}
	if end != "" && ts >= end {
		return false
	}
	return true
}

func encode(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
