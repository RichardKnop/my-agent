// Package claude is a deliberately thin, dependency-free client for
// Anthropic's Messages API. There's no SDK magic here on purpose — you can
// see exactly what a tool-use request/response looks like on the wire,
// which is the part most tutorials hide behind an SDK object.
package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

const (
	defaultBaseURL   = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
)

type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewClient builds a client for any server that speaks the Anthropic
// Messages API. baseURL may be empty (Anthropic's hosted API), a full
// ".../v1/messages" URL, or just a host root such as
// "http://localhost:11434" (Ollama), in which case "/v1/messages" is
// appended — the same convention as the ANTHROPIC_BASE_URL env var.
//
// apiKey may be empty for local servers that don't check it; the
// x-api-key header is then omitted.
func NewClient(apiKey, baseURL string) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: ResolveURL(baseURL),
		// Local models on a laptop/desktop can take well over a minute
		// for one turn once the conversation grows, so the per-request
		// timeout is generous; the caller's context bounds the whole run.
		httpClient: &http.Client{Timeout: 5 * time.Minute},
	}
}

// ResolveURL normalises a base URL into the full Messages endpoint.
func ResolveURL(baseURL string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return defaultBaseURL
	}
	if strings.HasSuffix(baseURL, "/v1/messages") {
		return baseURL
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + "/messages"
	}
	return baseURL + "/v1/messages"
}

// IsAnthropicHosted reports whether the resolved URL is Anthropic's own
// API (where a real key is required) rather than a local/compatible server.
func IsAnthropicHosted(baseURL string) bool {
	return strings.HasPrefix(ResolveURL(baseURL), "https://api.anthropic.com/")
}

// ContentBlock covers every block type this project uses: assistant text,
// assistant tool_use requests, and user tool_result replies. Anthropic's
// real schema has more block types (images, thinking, etc.) — this is the
// subset a text-and-tools agent needs.
type ContentBlock struct {
	Type string `json:"type"`

	// type == "text"
	Text string `json:"text,omitempty"`

	// type == "tool_use" (assistant asking to call a tool)
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// type == "tool_result" (user turn, giving a tool's output back)
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
}

type Message struct {
	Role    string         `json:"role"` // "user" | "assistant"
	Content []ContentBlock `json:"content"`
}

// Tool is a tool definition as the Messages API expects it: a name, a
// description the model reads to decide when to call it, and a JSON Schema
// for its input.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type Request struct {
	Model     string    `json:"model"`
	MaxTokens int       `json:"max_tokens"`
	System    string    `json:"system,omitempty"`
	Messages  []Message `json:"messages"`
	Tools     []Tool    `json:"tools,omitempty"`
}

type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

type Response struct {
	ID         string         `json:"id"`
	Role       string         `json:"role"`
	Content    []ContentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      Usage          `json:"usage"`
}

type apiError struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// apiCallError carries the HTTP status code alongside the message so
// CreateMessage can decide whether it's worth retrying.
type apiCallError struct {
	statusCode int
	message    string
}

func (e *apiCallError) Error() string {
	return fmt.Sprintf("messages api error (status %d): %s", e.statusCode, e.message)
}

func (e *apiCallError) retryable() bool {
	return e.statusCode == http.StatusTooManyRequests || e.statusCode >= 500
}

// CreateMessage sends one Messages API request, retrying on 429 and 5xx
// with exponential backoff — the kind of thing that's invisible in a demo
// but is the difference between an agent that's flaky in production and
// one that isn't.
func (c *Client) CreateMessage(ctx context.Context, req Request) (*Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	const maxAttempts = 4
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt))) * time.Second
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		resp, err := c.doOnce(ctx, body)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		var ce *apiCallError
		if !errors.As(err, &ce) || !ce.retryable() {
			return nil, err
		}
	}
	return nil, fmt.Errorf("giving up after %d attempts: %w", maxAttempts, lastErr)
}

func (c *Client) doOnce(ctx context.Context, body []byte) (*Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("content-type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("x-api-key", c.apiKey)
	}
	httpReq.Header.Set("anthropic-version", anthropicVersion)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		var ae apiError
		_ = json.Unmarshal(respBody, &ae)
		msg := ae.Error.Message
		if msg == "" {
			msg = string(respBody)
		}
		return nil, &apiCallError{statusCode: httpResp.StatusCode, message: msg}
	}

	var out Response
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	return &out, nil
}
