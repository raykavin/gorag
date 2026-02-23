package engine

import (
	"context"

	"github.com/tmc/langchaingo/schema"
)

// Default chunk constants
const (
	defTextChunkSize    = 1200
	defTextChunkOverlap = 180
	defCSVRowsPerChunk  = 20
)

// Default embed constants
const (
	defEmbedWorkers   = 10
	defEmbedBatchSize = 16
)

// DocumentReranker reranks a set of candidate documents for a given query.
type DocumentReranker interface {
	Rerank(ctx context.Context, query string, docs []schema.Document, topK int) ([]schema.Document, error)
}

// batch holds the original indexes
// and corresponding texts for one parallel unit of work.
type batch struct {
	indexes []int
	texts   []string
}

// rerankRequest is the JSON payload sent to the rerank API.
type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// rerankResponse accepts both "results" and "data" field names for compatibility.
type rerankResponse struct {
	Results []rerankResult `json:"results"`
	Data    []rerankResult `json:"data"`
}

// rerankResult represents a single scored entry from the reranker.
type rerankResult struct {
	Index          int     `json:"index"`
	Score          float64 `json:"score"`
	RelevanceScore float64 `json:"relevance_score"`
}

// rerankScore returns the best available score from a rerank result.
func rerankScore(result rerankResult) float64 {
	if result.RelevanceScore != 0 {
		return result.RelevanceScore
	}
	return result.Score
}
