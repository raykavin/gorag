package engine

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"

	"github.com/raykavin/gorag/cache"
)

// RetrievalMode defines the document retrieval strategy.
type RetrievalMode string

const (
	RetrievalDense    RetrievalMode = "dense"
	RetrievalReranker RetrievalMode = "reranker"

	defaultTopK     = 3
	defaultMinScore = float32(0.7)
)

// RAGEngine retrieves relevant document chunks from a vector store
// and optionally reranks them before injecting into a prompt.
type RAGEngine struct {
	store         vectorstores.VectorStore
	reranker      DocumentReranker
	embCache      *cache.Embedding
	topK          int
	minScore      float32
	forceAlways   bool
	retrievalMode RetrievalMode
	rerankerK     int
	chunksLoaded  int
}

// Config holds the configuration parameters for a RAGEngine.
type Config struct {
	TopK          int
	MinScore      float32
	ForceAlways   bool
	RetrievalMode string
	RerankerK     int
	ChunksLoaded  int
}

// NewRAGEngine creates a RAGEngine with the given store, reranker, cache and config.
func NewRAGEngine(
	store vectorstores.VectorStore,
	reranker DocumentReranker,
	embCache *cache.Embedding,
	cfg Config,
) *RAGEngine {
	if cfg.TopK <= 0 {
		cfg.TopK = defaultTopK
	}
	if cfg.MinScore <= 0 {
		cfg.MinScore = defaultMinScore
	}
	if cfg.RerankerK < cfg.TopK {
		cfg.RerankerK = cfg.TopK
	}
	return &RAGEngine{
		store:         store,
		reranker:      reranker,
		embCache:      embCache,
		topK:          cfg.TopK,
		minScore:      cfg.MinScore,
		forceAlways:   cfg.ForceAlways,
		retrievalMode: normalizeRetrievalMode(cfg.RetrievalMode),
		rerankerK:     cfg.RerankerK,
		chunksLoaded:  cfg.ChunksLoaded,
	}
}

// Query retrieves relevant context for the given query and returns it
// formatted as a prompt block. The boolean return indicates whether any
// context was found.
func (r *RAGEngine) Query(ctx context.Context, query string) (string, bool, error) {
	if r == nil {
		return "", false, nil
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return "", false, nil
	}

	results, err := r.retrieve(ctx, query)
	if err != nil {
		return "", false, err
	}

	lines := make([]string, 0, len(results))
	for _, res := range results {
		if content := strings.TrimSpace(res.PageContent); content != "" {
			lines = append(lines, content)
		}
	}
	if len(lines) == 0 {
		return "", false, nil
	}

	return buildContextBlock(lines), true, nil
}

// ChunksLoaded returns the number of knowledge chunks loaded into the store.
func (r *RAGEngine) ChunksLoaded() int {
	if r == nil {
		return 0
	}
	return r.chunksLoaded
}

// EmbeddingCacheStats returns cache statistics for the embedding cache.
func (r *RAGEngine) EmbeddingCacheStats() cache.CacheStats {
	if r == nil || r.embCache == nil {
		return cache.CacheStats{}
	}
	return r.embCache.Stats()
}

// retrieve dispatches to the appropriate retrieval strategy.
func (r *RAGEngine) retrieve(ctx context.Context, query string) ([]schema.Document, error) {
	switch r.retrievalMode {
	case RetrievalReranker:
		return r.retrieveWithReranker(ctx, query)
	default:
		return r.searchDense(ctx, query, r.topK)
	}
}

// retrieveWithReranker performs dense retrieval followed by reranking.
func (r *RAGEngine) retrieveWithReranker(ctx context.Context, query string) ([]schema.Document, error) {
	if r.reranker == nil {
		return nil, fmt.Errorf("reranker mode is active but no reranker was provided")
	}

	candidates, err := r.searchDenseCandidates(ctx, query)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	reranked, err := r.reranker.Rerank(ctx, query, candidates, r.topK)
	if err != nil {
		return nil, fmt.Errorf("reranker failed: %w", err)
	}
	return reranked, nil
}

// searchDense performs a similarity search with an optional score threshold.
// When forceAlways is true and no results pass the threshold, it retries without one.
func (r *RAGEngine) searchDense(ctx context.Context, query string, topK int) ([]schema.Document, error) {
	if r.store == nil {
		return nil, nil
	}
	if topK <= 0 {
		topK = r.topK
	}

	results, err := r.store.SimilaritySearch(ctx, query, topK, vectorstores.WithScoreThreshold(r.minScore))
	if err != nil {
		return nil, fmt.Errorf("semantic search failed: %w", err)
	}

	if len(results) == 0 && r.forceAlways {
		// Fallback without threshold to always return some context.
		results, err = r.store.SimilaritySearch(ctx, query, topK)
		if err != nil {
			return nil, fmt.Errorf("semantic search fallback failed: %w", err)
		}
	}
	return results, nil
}

// searchDenseCandidates retrieves rerankerK candidates, falling back to a no-threshold
// search to improve recall when the scored search returns too few results.
func (r *RAGEngine) searchDenseCandidates(ctx context.Context, query string) ([]schema.Document, error) {
	candidates, err := r.searchDense(ctx, query, r.rerankerK)
	if err != nil {
		return nil, err
	}
	if len(candidates) >= r.rerankerK || r.store == nil {
		return candidates, nil
	}

	fallback, err := r.store.SimilaritySearch(ctx, query, r.rerankerK)
	if err != nil {
		return nil, fmt.Errorf("candidate search fallback failed: %w", err)
	}
	return mergeUniqueDocuments(candidates, fallback, r.rerankerK), nil
}

// buildContextBlock formats retrieved lines into a prompt-ready context block.
func buildContextBlock(lines []string) string {
	var b strings.Builder
	_, _ = b.WriteString("## Relevant Context\n")
	for i, line := range lines {
		_, _ = b.WriteString(line)
		if i < len(lines)-1 {
			_, _ = b.WriteString("\n\n")
		}
	}
	return b.String()
}

// mergeUniqueDocuments merges two document slices, deduplicating by key, up to topK results.
func mergeUniqueDocuments(first, second []schema.Document, topK int) []schema.Document {
	capacity := len(first) + len(second)
	if topK > 0 && topK < capacity {
		capacity = topK
	}

	out := make([]schema.Document, 0, capacity)
	seen := make(map[string]struct{}, len(first)+len(second))

	for _, batch := range [][]schema.Document{first, second} {
		for _, doc := range batch {
			if topK > 0 && len(out) >= topK {
				return out
			}
			if content := strings.TrimSpace(doc.PageContent); content == "" {
				continue
			}
			key := documentKey(doc)
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, doc)
		}
	}
	return out
}

// documentKey returns a stable deduplication key for a document,
// preferring metadata "id", then "source", then falling back to content.
func documentKey(doc schema.Document) string {
	if doc.Metadata != nil {
		if id, ok := metadataString(doc.Metadata, "id"); ok {
			return "id:" + id
		}
		if source, ok := metadataString(doc.Metadata, "source"); ok {
			return "source:" + source
		}
	}
	return "content:" + strings.TrimSpace(doc.PageContent)
}

// metadataString extracts a non-empty string value from a metadata map.
func metadataString(meta map[string]any, key string) (string, bool) {
	raw, ok := meta[key]
	if !ok {
		return "", false
	}
	s, ok := raw.(string)
	if !ok {
		return "", false
	}
	s = strings.TrimSpace(s)
	return s, s != ""
}

// normalizeRetrievalMode maps a raw string to a RetrievalMode, defaulting to dense.
func normalizeRetrievalMode(mode string) RetrievalMode {
	switch RetrievalMode(strings.ToLower(strings.TrimSpace(mode))) {
	case RetrievalReranker:
		return RetrievalReranker
	default:
		return RetrievalDense
	}
}
