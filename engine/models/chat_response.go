package models

// ChatResponse is used here for simplicity as the cache package
// handles the storage of the final engine output
type ChatResponse struct {
	Output     string        `json:"output"`
	ToolCalls  []ToolCallLog `json:"tool_calls"`
	RAGUsed    bool          `json:"rag_used"`
	CacheHit   bool          `json:"cache_hit"`
	Latency    int64         `json:"latency_ms"`
	TokensUsed int           `json:"tokens_used"`
	Error      *string       `json:"error,omitempty"`
}
