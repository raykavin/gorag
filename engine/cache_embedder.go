package engine

import (
	"context"
	"errors"
	"gorag/engine/cache"

	"github.com/tmc/langchaingo/embeddings"
)

// CacheEmbedder is a wrapper around an embeddings.Embedder that caches results.
type CacheEmbedder struct {
	base  embeddings.Embedder
	cache *cache.Embedding
}

var (
	ErrInvalidCache                    = errors.New("invalid cache")
	ErrInvalidEmbeddingBase            = errors.New("invalid embedding base")
	ErrInvalidReturnableVectorQuantity = errors.New("invalid returnable vector quantity")
)

// NewCacheEmbedder creates a new CacheEmbedder.
func NewCacheEmbedder(
	base embeddings.Embedder,
	cache *cache.Embedding,
) (*CacheEmbedder, error) {
	if cache == nil {
		return nil, ErrInvalidCache
	}
	if base == nil {
		return nil, ErrInvalidEmbeddingBase
	}

	return &CacheEmbedder{
		base:  base,
		cache: cache,
	}, nil
}

// EmbedQuery embeds a query
func (e *CacheEmbedder) EmbedQuery(
	ctx context.Context,
	text string,
) ([]float32, error) {
	if e == nil || e.base == nil {
		return nil, ErrInvalidEmbeddingBase
	}
	if cached, ok := e.cache.Get(text); ok {
		return cached, nil
	}

	// Embeds a query
	emb, err := e.base.EmbedQuery(ctx, text)
	if err != nil {
		return nil, err
	}

	// Cache the embedding
	e.cache.Set(text, emb)
	return emb, nil
}

func (e *CacheEmbedder) EmbedDocuments(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	if e == nil || e.base == nil {
		return nil, ErrInvalidEmbeddingBase
	}

	lenTexts := len(texts)

	if lenTexts == 0 {
		return [][]float32{}, nil
	}

	// Initialize the result slice and tracking for cache misses
	vectors := make([][]float32, lenTexts)
	missingTexts := make([]string, 0, lenTexts)
	missingIndexes := make([]int, 0, lenTexts)

	// Check cache for each text
	for i, text := range texts {
		if cached, ok := e.cache.Get(text); ok {
			vectors[i] = cached
			continue
		}
		missingTexts = append(missingTexts, text)
		missingIndexes = append(missingIndexes, i)
	}

	if len(missingTexts) == 0 {
		return vectors, nil
	}

	// Embed the missing texts
	generated, err := e.base.EmbedDocuments(ctx, missingTexts)
	if err != nil {
		return nil, err
	}
	if len(generated) != len(missingTexts) {
		return nil, ErrInvalidReturnableVectorQuantity
	}

	// Update the vectors slice and cache the new embeddings
	for i, vec := range generated {
		idx := missingIndexes[i]
		vectors[idx] = append([]float32(nil), vec...)
		e.cache.Set(missingTexts[idx], vec)
	}

	return vectors, nil
}
