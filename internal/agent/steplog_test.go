package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/richardknop/my-agent/internal/claude"
)

func TestTruncationSuspected(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		payload int
		usage   claude.Usage
		want    bool
	}{
		{
			name:    "full prompt counted, ~3 bytes/token",
			payload: 4934,
			usage:   claude.Usage{InputTokens: 1510},
			want:    false,
		},
		{
			// Turn 5 of a real Ollama run: only the newly appended tokens
			// are input_tokens, the reused prefix is cache_read. This used
			// to trigger a false warning on every turn after the first.
			name:    "Ollama prefix cache",
			payload: 25357,
			usage:   claude.Usage{InputTokens: 260, CacheReadInputTokens: 13300},
			want:    false,
		},
		{
			name:    "far too few tokens for the payload",
			payload: 25357,
			usage:   claude.Usage{InputTokens: 260},
			want:    true,
		},
		{
			name:    "no usage reported",
			payload: 25357,
			want:    false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, got := truncationSuspected(tc.payload, tc.usage)
			assert.Equal(t, tc.want, got)
		})
	}
}
