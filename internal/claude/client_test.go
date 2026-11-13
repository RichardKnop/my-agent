package claude

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveURL(t *testing.T) {
	cases := map[string]string{
		"":                                   defaultBaseURL,
		"http://localhost:11434":             "http://localhost:11434/v1/messages",
		"http://localhost:11434/":            "http://localhost:11434/v1/messages",
		"http://localhost:11434/v1":          "http://localhost:11434/v1/messages",
		"http://localhost:11434/v1/messages": "http://localhost:11434/v1/messages",
	}
	for in, want := range cases {
		assert.Equal(t, want, ResolveURL(in), "ResolveURL(%q)", in)
	}
	assert.True(t, IsAnthropicHosted(""))
	assert.False(t, IsAnthropicHosted("http://localhost:11434"))
}

// A local server (e.g. Ollama) gets the same request shape, without an API
// key header when none is configured, and its tool_use reply decodes.
func TestCreateMessageAgainstLocalServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// assert, not require: require calls t.FailNow, which must not run
		// outside the test goroutine (this handler runs on the server's).
		assert.Equal(t, "/v1/messages", r.URL.Path)
		assert.Empty(t, r.Header.Get("x-api-key"), "x-api-key must not be sent to a local server")

		var req Request
		if assert.NoError(t, json.NewDecoder(r.Body).Decode(&req)) {
			assert.Equal(t, "qwen3-coder", req.Model)
		}

		w.Header().Set("content-type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","role":"assistant","stop_reason":"tool_use",
			"content":[{"type":"tool_use","id":"call_1","name":"list_services","input":{}}],
			"usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer srv.Close()

	c := NewClient("", srv.URL)
	resp, err := c.CreateMessage(context.Background(), Request{
		Model: "qwen3-coder", MaxTokens: 100,
		Messages: []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
	})

	require.NoError(t, err)
	assert.Equal(t, "tool_use", resp.StopReason)
	require.Len(t, resp.Content, 1)
	assert.Equal(t, "list_services", resp.Content[0].Name)
}
