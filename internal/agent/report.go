package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/richardknop/my-agent/internal/claude"
)

// The agent's answer is a structured Report, submitted by calling the
// submit_report tool, rather than free text. This is the standard way to get
// structured output from a tool-calling model: describe the output as a
// tool's input schema, and the model "calls" the tool to answer.
//
// Why bother: free text can only be graded by matching phrases (which breaks
// when a different model words things differently) or by another model
// (which costs money and varies). Fields like root_cause_service can be
// compared exactly, which is what the evals stage needs.
//
// submit_report is not in the tools.Registry on purpose. Registry tools are
// the agent's view of the system (and will move behind an MCP server in
// stage 3); submit_report is part of the loop's own protocol: calling it ends
// the run.

const submitReportToolName = "submit_report"

// Report is the agent's final answer.
type Report struct {
	Summary           string   `json:"summary"`
	RootCauseService  string   `json:"root_cause_service"`
	TriggeringChange  string   `json:"triggering_change"`
	AffectedServices  []string `json:"affected_services"`
	Evidence          []string `json:"evidence"`
	Confidence        string   `json:"confidence"`
	ConfidenceReason  string   `json:"confidence_reason"`
	RecommendedAction string   `json:"recommended_action"`
}

// The field descriptions are prompt text: the model reads them to decide
// what goes where. They must stay neutral about the answer. Nothing here
// may suggest that the cause is a deploy, or that there is a cause at all,
// or the evals stop measuring the investigation.
var submitReportTool = claude.Tool{
	Name: submitReportToolName,
	Description: "Submit your final incident report. This is the only way to answer: " +
		"plain-text answers are not recorded. Calling it ends the investigation, so call it " +
		"once, after gathering evidence with the other tools, and not together with them. " +
		"If the input is invalid you get an error back and can call it again.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"summary": map[string]any{
				"type":        "string",
				"description": "The root cause and its mechanism in one or two sentences.",
			},
			"root_cause_service": map[string]any{
				"type": "string",
				"description": "Name of the service where the problem originated, exactly as " +
					"list_services returns it. Empty string if you found no incident.",
			},
			"triggering_change": map[string]any{
				"type": "string",
				"description": "Identifier of the change that triggered the problem, exactly as " +
					"it appears in the data, if one did. Empty string if no change triggered it " +
					"or you could not identify one.",
			},
			"affected_services": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Services that showed impact, including the root-cause service. " +
					"A JSON array with one service name per element.",
			},
			"evidence": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
				"description": "Specific data points supporting the conclusion: timestamps, " +
					"services, metric values, deploy versions or log messages. A JSON array with " +
					"one string per data point, not a single bulleted string.",
			},
			"confidence": map[string]any{
				"type": "string",
				"enum": []string{"low", "medium", "high"},
			},
			"confidence_reason": map[string]any{
				"type":        "string",
				"description": "Why this confidence level; mention any evidence that is missing.",
			},
			"recommended_action": map[string]any{
				"type":        "string",
				"description": "What to do next, e.g. a rollback, an alert, a follow-up query.",
			},
		},
		"required": []string{
			"summary", "root_cause_service", "triggering_change", "affected_services",
			"evidence", "confidence", "confidence_reason", "recommended_action",
		},
	},
}

// parseReport decodes and checks a submit_report input. The schema already
// says all this, but models don't always follow schemas (and servers such as
// Ollama don't enforce them), so the loop checks too and sends any problem
// back to the model as a tool error it can fix.
func parseReport(input json.RawMessage) (*Report, error) {
	var r Report
	if err := json.Unmarshal(input, &r); err != nil {
		return nil, errors.New("invalid report: " + describeDecodeError(err))
	}
	var problems []string
	if strings.TrimSpace(r.Summary) == "" {
		problems = append(problems, "summary is required")
	}
	r.Confidence = strings.ToLower(strings.TrimSpace(r.Confidence))
	switch r.Confidence {
	case "low", "medium", "high":
	default:
		problems = append(problems, fmt.Sprintf("confidence must be one of low, medium, high (got %q)", r.Confidence))
	}
	if len(problems) > 0 {
		return nil, errors.New("invalid report: " + strings.Join(problems, "; "))
	}
	return &r, nil
}

// describeDecodeError turns a JSON decoding error into an instruction the
// model can act on. Go's own message ("cannot unmarshal string into Go struct
// field Report.evidence of type []string") names Go types and structs the
// model has never seen; a retry is more likely to succeed when the error says
// what the field should look like.
func describeDecodeError(err error) string {
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) && typeErr.Field != "" {
		field, _, _ := strings.Cut(typeErr.Field, ".")
		want := "a string"
		if prop, ok := submitReportTool.InputSchema["properties"].(map[string]any)[field].(map[string]any); ok && prop["type"] == "array" {
			want = `an array of strings, e.g. ["first item", "second item"], not one string`
		}
		return fmt.Sprintf("%s must be %s (got a JSON %s)", field, want, typeErr.Value)
	}
	return "the input is not a valid JSON object: " + err.Error()
}

// Text renders the report for a human, in the sections the agent used to
// write as free text.
func (r *Report) Text() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Root cause: %s\n", r.Summary)
	fmt.Fprintf(&sb, "Root-cause service: %s\n", orNone(r.RootCauseService))
	fmt.Fprintf(&sb, "Triggering change: %s\n", orNone(r.TriggeringChange))
	fmt.Fprintf(&sb, "Affected services: %s\n", orNone(strings.Join(r.AffectedServices, ", ")))
	sb.WriteString("Evidence:\n")
	if len(r.Evidence) == 0 {
		sb.WriteString("  (none given)\n")
	}
	for _, e := range r.Evidence {
		fmt.Fprintf(&sb, "  - %s\n", e)
	}
	fmt.Fprintf(&sb, "Confidence: %s", r.Confidence)
	if r.ConfidenceReason != "" {
		fmt.Fprintf(&sb, " (%s)", r.ConfidenceReason)
	}
	fmt.Fprintf(&sb, "\nRecommended action: %s\n", orNone(r.RecommendedAction))
	return sb.String()
}

func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "none identified"
	}
	return s
}
