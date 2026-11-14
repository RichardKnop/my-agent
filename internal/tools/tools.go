// Package tools implements the small set of read-only tools the agent can
// call: list_services, query_metrics, query_logs and get_deploys. Each tool
// has a Go implementation plus a JSON-schema Definition. The agent sends the
// definitions to the model in the `tools` field of every Messages API
// request; an MCP server (roadmap stage 3 in README) would expose the same
// definitions over its own protocol.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/richardknop/my-agent/internal/data"
)

// Definition describes one tool independently of any model API or
// protocol: a name, a description the model uses to decide *when* to call
// it, and a JSON Schema for its input. The agent converts it into the
// Messages API's tool type; an MCP server (stage 3) would convert it into
// MCP's.
type Definition struct {
	Name        string
	Description string
	InputSchema map[string]any
}

// Handler executes one tool call and returns the result already
// JSON-encoded, ready to go back to the model as a tool_result. ctx carries
// the run's deadline, so a slow tool (a real database query, a remote MCP
// server) can be cancelled with the rest of the run.
type Handler func(ctx context.Context, input json.RawMessage) (string, error)

// Registry pairs each tool's definition with its handler so the agent loop
// can look either up by name without a big switch statement. The handlers
// are bound to their dependencies (here, the dataset) when the registry is
// built, so callers only ever deal in tool names and JSON input.
type Registry struct {
	defs     []Definition
	handlers map[string]Handler
}

// NewRegistry builds the registry of observability tools over store.
func NewRegistry(store *data.Store) *Registry {
	r := &Registry{handlers: map[string]Handler{}}
	r.register(listServicesDef, withStore(store, listServices))
	r.register(queryMetricsDef, withStore(store, queryMetrics))
	r.register(queryLogsDef, withStore(store, queryLogs))
	r.register(getDeploysDef, withStore(store, getDeploys))
	return r
}

func (r *Registry) register(def Definition, h Handler) {
	r.defs = append(r.defs, def)
	r.handlers[def.Name] = h
}

func (r *Registry) Definitions() []Definition { return r.defs }

func (r *Registry) Call(ctx context.Context, name string, input json.RawMessage) (string, error) {
	h, ok := r.handlers[name]
	if !ok {
		return "", fmt.Errorf("unknown tool %q", name)
	}
	return h(ctx, input)
}

// withStore adapts a tool function over the in-memory dataset into a
// Handler. The dataset is in memory, so these tools have nothing to cancel
// and ignore ctx.
func withStore(store *data.Store, fn func(*data.Store, json.RawMessage) (string, error)) Handler {
	return func(_ context.Context, input json.RawMessage) (string, error) {
		return fn(store, input)
	}
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

// query_logs has two output modes, and the default is the interesting
// context-engineering decision.
//
// "grouped" (default) collapses identical lines into one row per
// (service, level, message) with a count, first/last seen timestamps and a
// few sample trace_ids. Incident logs are extremely repetitive: in the mock
// dataset 1,663 ERROR lines are just 3 distinct messages. Returning the 40
// newest raw lines would show the model 40 copies of one message from the
// tail of the incident and cut off the two things it needs most: when the
// errors *started*, and rare one-off lines such as a deploy notice. Grouped
// rows keep both, in a fraction of the tokens.
//
// "raw" returns individual lines, newest first, for when the model needs
// them, typically to follow one request across services with trace_id.
//
// Grouping is by exact message. Real logs embed variable parts (request IDs,
// durations, counters), so production systems first turn messages into
// templates by masking those parts (e.g. the Drain algorithm). Our mock
// messages are constant, so exact matching is enough here.
const (
	maxLogResults     = 40 // raw mode: max lines returned
	maxLogGroups      = 40 // grouped mode: max groups returned
	maxSampleTraceIDs = 3  // grouped mode: trace_ids kept per group
	logModeGrouped    = "grouped"
	logModeRaw        = "raw"
)

var queryLogsDef = Definition{
	Name: "query_logs",
	Description: fmt.Sprintf(
		"Search log lines within a time window, optionally filtered by service, level, trace_id "+
			"and a substring match on the message.\n\n"+
			"By default (mode \"grouped\") identical lines are collapsed into one group per "+
			"(service, level, message), with the number of occurrences, the first_seen and last_seen "+
			"timestamps, and up to %d sample trace_ids. Groups are ordered by first_seen, so they read "+
			"as a timeline of when each kind of message started. At most %d groups are returned.\n\n"+
			"Use mode \"raw\" to get individual lines instead (most recent first, at most %d). "+
			"To follow one request across services, take a trace_id from a group's sample_trace_ids "+
			"and query with mode \"raw\" and that trace_id, without a service filter.\n\n"+
			"If the result says truncated, narrow the time window or add filters.",
		maxSampleTraceIDs, maxLogGroups, maxLogResults),
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
			"trace_id": map[string]any{
				"type":        "string",
				"description": "Optional exact trace_id; returns only lines from that one request.",
			},
			"mode": map[string]any{
				"type":        "string",
				"enum":        []string{logModeGrouped, logModeRaw},
				"description": "\"grouped\" (default) collapses identical lines with counts; \"raw\" returns individual lines.",
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
	TraceID  string `json:"trace_id"`
	Mode     string `json:"mode"`
	Start    string `json:"start"`
	End      string `json:"end"`
}

// LogGroup is one row of grouped query_logs output.
type LogGroup struct {
	Service        string   `json:"service"`
	Level          string   `json:"level"`
	Message        string   `json:"message"`
	Count          int      `json:"count"`
	FirstSeen      string   `json:"first_seen"`
	LastSeen       string   `json:"last_seen"`
	SampleTraceIDs []string `json:"sample_trace_ids,omitempty"`
}

func queryLogs(store *data.Store, input json.RawMessage) (string, error) {
	var a logsArgs
	if err := json.Unmarshal(input, &a); err != nil {
		return "", fmt.Errorf("bad input: %w", err)
	}
	if a.Mode == "" {
		a.Mode = logModeGrouped
	}
	if a.Mode != logModeGrouped && a.Mode != logModeRaw {
		return "", fmt.Errorf("unknown mode %q: use %q or %q", a.Mode, logModeGrouped, logModeRaw)
	}

	matched := filterLogs(store.Logs, a)

	if a.Mode == logModeRaw {
		sort.Slice(matched, func(i, j int) bool { return matched[i].TS > matched[j].TS })
		total := len(matched)
		truncated := total > maxLogResults
		if truncated {
			matched = matched[:maxLogResults]
		}
		return encode(map[string]any{
			"mode":          logModeRaw,
			"total_matches": total,
			"returned":      len(matched),
			"truncated":     truncated,
			"entries":       matched,
		})
	}

	groups := groupLogs(matched)
	totalGroups := len(groups)
	truncated := totalGroups > maxLogGroups
	if truncated {
		groups = groups[:maxLogGroups]
	}
	return encode(map[string]any{
		"mode":            logModeGrouped,
		"total_matches":   len(matched),
		"total_groups":    totalGroups,
		"returned_groups": len(groups),
		"truncated":       truncated,
		"groups":          groups,
	})
}

func filterLogs(logs []data.LogEntry, a logsArgs) []data.LogEntry {
	contains := strings.ToLower(a.Contains)
	var matched []data.LogEntry
	for _, l := range logs {
		if !inRange(l.TS, a.Start, a.End) {
			continue
		}
		if a.Service != "" && l.Service != a.Service {
			continue
		}
		if a.Level != "" && l.Level != a.Level {
			continue
		}
		if a.TraceID != "" && l.TraceID != a.TraceID {
			continue
		}
		if contains != "" && !strings.Contains(strings.ToLower(l.Message), contains) {
			continue
		}
		matched = append(matched, l)
	}
	return matched
}

// groupLogs collapses entries with the same (service, level, message) and
// returns the groups ordered by first_seen, so they read as a timeline.
// Sample trace_ids are the earliest distinct ones, which point at requests
// from around the moment a problem started.
func groupLogs(entries []data.LogEntry) []LogGroup {
	sorted := slices.Clone(entries)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].TS < sorted[j].TS })

	type key struct{ service, level, message string }
	index := map[key]int{}
	var groups []LogGroup
	for _, e := range sorted {
		k := key{e.Service, e.Level, e.Message}
		i, ok := index[k]
		if !ok {
			i = len(groups)
			index[k] = i
			groups = append(groups, LogGroup{
				Service: e.Service, Level: e.Level, Message: e.Message, FirstSeen: e.TS,
			})
		}
		g := &groups[i]
		g.Count++
		g.LastSeen = e.TS
		if e.TraceID != "" && len(g.SampleTraceIDs) < maxSampleTraceIDs && !slices.Contains(g.SampleTraceIDs, e.TraceID) {
			g.SampleTraceIDs = append(g.SampleTraceIDs, e.TraceID)
		}
	}
	// Groups were created in first_seen order already; make ties
	// deterministic so output doesn't depend on input order.
	sort.SliceStable(groups, func(i, j int) bool {
		if groups[i].FirstSeen != groups[j].FirstSeen {
			return groups[i].FirstSeen < groups[j].FirstSeen
		}
		if groups[i].Service != groups[j].Service {
			return groups[i].Service < groups[j].Service
		}
		return groups[i].Message < groups[j].Message
	})
	return groups
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
