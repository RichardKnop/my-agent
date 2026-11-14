package agent

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseReport(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		input   string
		wantErr []string // substrings; empty = valid
	}{
		{name: "valid", input: validReport},
		{
			name:  "no incident found is a valid report",
			input: `{"summary": "no incident in the window", "root_cause_service": "", "triggering_change": "", "confidence": "medium"}`,
		},
		{
			name:  "confidence is normalised",
			input: `{"summary": "x", "confidence": " High "}`,
		},
		{
			name:    "missing summary and bad confidence reported together",
			input:   `{"summary": "  ", "confidence": "certain"}`,
			wantErr: []string{"summary is required", `confidence must be one of low, medium, high (got "certain")`},
		},
		{
			name:    "missing confidence",
			input:   `{"summary": "x"}`,
			wantErr: []string{`confidence must be one of low, medium, high (got "")`},
		},
		{
			name:    "wrong type",
			input:   `{"summary": "x", "confidence": "low", "evidence": "one string, not a list"}`,
			wantErr: []string{"invalid report", "evidence"},
		},
		{
			name:    "malformed JSON",
			input:   `{"summary": `,
			wantErr: []string{"invalid report"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			r, err := parseReport(json.RawMessage(tc.input))

			if len(tc.wantErr) == 0 {
				require.NoError(t, err)
				assert.Contains(t, []string{"low", "medium", "high"}, r.Confidence)
				return
			}
			require.Error(t, err)
			assert.Nil(t, r)
			for _, want := range tc.wantErr {
				assert.Contains(t, err.Error(), want)
			}
		})
	}
}

func TestReportText(t *testing.T) {
	t.Parallel()

	r, err := parseReport(json.RawMessage(validReport))
	require.NoError(t, err)

	assert.Equal(t, `Root cause: payments rejected connections after a config change
Root-cause service: payments
Triggering change: v9.9.9
Affected services: payments, checkout
Evidence:
  - checkout error_rate 0.4 at 14:35
Confidence: high (deploy lines up with the errors)
Recommended action: roll back
`, r.Text())

	empty := &Report{Summary: "nothing found", Confidence: "low"}
	assert.Equal(t, `Root cause: nothing found
Root-cause service: none identified
Triggering change: none identified
Affected services: none identified
Evidence:
  (none given)
Confidence: low
Recommended action: none identified
`, empty.Text())
}

// The schema's required list and properties must match the Report fields,
// or the model is asked for fields we drop (or never asked for ones we read).
func TestSubmitReportSchemaMatchesReport(t *testing.T) {
	t.Parallel()

	b, err := json.Marshal(Report{})
	require.NoError(t, err)
	var fields map[string]any
	require.NoError(t, json.Unmarshal(b, &fields))

	props := submitReportTool.InputSchema["properties"].(map[string]any)
	required := submitReportTool.InputSchema["required"].([]string)
	assert.Len(t, props, len(fields))
	assert.Len(t, required, len(fields))
	for name := range fields {
		assert.Contains(t, props, name)
		assert.Contains(t, required, name)
	}
}
