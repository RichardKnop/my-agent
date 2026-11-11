package claude

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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
		if got := ResolveURL(in); got != want {
			t.Errorf("ResolveURL(%q) = %q, want %q", in, got, want)
		}
	}
	if !IsAnthropicHosted("") || IsAnthropicHosted("http://localhost:11434") {
		t.Error("IsAnthropicHosted misclassified a URL")
	}
}

// A local server (e.g. Ollama) gets the same request shape, without an API
// key header when none is configured, and its tool_use reply decodes.
func TestCreateMessageAgainstLocalServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %q, want /v1/messages", r.URL.Path)
		}
		if k := r.Header.Get("x-api-key"); k != "" {
			t.Errorf("x-api-key sent to local server: %q", k)
		}
		var req Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "qwen3-coder" {
			t.Errorf("model = %q", req.Model)
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
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != "tool_use" || len(resp.Content) != 1 || resp.Content[0].Name != "list_services" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}
