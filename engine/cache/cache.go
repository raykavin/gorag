package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"gorag/engine/models"
	"regexp"
	"strings"
)

var (
	irrelevantPunctuationPattern = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)
	multiSpacePattern            = regexp.MustCompile(`\s+`)
)

// CacheStats represents the basic metrics of cache
type CacheStats struct {
	Hits    int64   `json:"hits"`
	Misses  int64   `json:"misses"`
	Size    int     `json:"size"`
	HitRate float64 `json:"hit_rate"`
}

// embeddingKey generates a unique hash for a text string to be used as a cache key
func embeddingKey(text string) string {
	h := sha256.Sum256([]byte(text))
	return hex.EncodeToString(h[:])
}

// buildResponseCacheKey generates a unique hash for the combination of text and history,
// returning both the hash and the normalized string used to generate it
func buildResponseCacheKey(
	text string,
	historyKey string,
) (string, string) {
	nText := normalizeText(text)
	nHistory := normalizeText(historyKey)
	joined := nText + nHistory

	h := sha256.Sum256([]byte(joined))
	return hex.EncodeToString(h[:]), joined
}

// normalizeText cleans the input string for consistent cache keys
func normalizeText(input string) string {
	if input == "" {
		return ""
	}

	lowered := strings.ToLower(strings.TrimSpace(input))
	clean := irrelevantPunctuationPattern.ReplaceAllString(lowered, " ")
	clean = multiSpacePattern.ReplaceAllString(clean, " ")
	return strings.TrimSpace(clean)
}

// cloneChatResponse creates a deep copy of the response to avoid data races
func cloneChatResponse(in *models.ChatResponse) *models.ChatResponse {
	if in == nil {
		return nil
	}
	out := *in
	if in.ToolCalls != nil {
		out.ToolCalls = append([]models.ToolCallLog(nil), in.ToolCalls...)
	}
	if in.Error != nil {
		errCopy := *in.Error
		out.Error = &errCopy
	}
	return &out
}
