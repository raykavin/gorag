package engine

import (
	"context"
	"fmt"
	"log/slog"
	"sync/atomic"

	"github.com/tmc/langchaingo/embeddings"
	"golang.org/x/sync/errgroup"
)

var (
	ErrNilEmbedder           = fmt.Errorf("base embedder cannot be nil")
	ErrUninitializedEmbedder = fmt.Errorf("progress embedder not initialized")
)

// ProgressEmbedder wraps an Embedder to process documents in parallel batches,
// logging progress as chunks are embedded.
type ProgressEmbedder struct {
	base      embeddings.Embedder
	total     int64
	workers   int
	batchSize int
	processed atomic.Int64
}

// NewProgressEmbedder creates a ProgressEmbedder with the given base embedder,
// expected total document count, and worker concurrency.
func NewProgressEmbedder(base embeddings.Embedder, total, workers int) (*ProgressEmbedder, error) {
	if base == nil {
		return nil, ErrNilEmbedder
	}
	if total < 0 {
		total = 0
	}
	if workers <= 0 {
		workers = defEmbedWorkers
	}
	return &ProgressEmbedder{
		base:      base,
		total:     int64(total),
		workers:   workers,
		batchSize: defEmbedBatchSize,
	}, nil
}

// EmbedQuery returns the embedding vector for a single query text.
func (e *ProgressEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if e == nil || e.base == nil {
		return nil, ErrUninitializedEmbedder
	}
	return e.base.EmbedQuery(ctx, text)
}

// EmbedDocuments embeds all texts in parallel batches, logging progress for each chunk.
func (e *ProgressEmbedder) EmbedDocuments(
	ctx context.Context,
	texts []string,
) ([][]float32, error) {
	if e == nil || e.base == nil {
		return nil, ErrUninitializedEmbedder
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	e.processed.Store(0)

	total := e.total
	if total <= 0 {
		total = int64(len(texts))
	}

	workers := e.workers
	if workers <= 0 {
		workers = defEmbedWorkers
	}
	if workers > len(texts) {
		workers = len(texts)
	}

	batchSize := e.batchSize
	if batchSize <= 0 {
		batchSize = defEmbedBatchSize
	}

	batches := buildBatches(texts, batchSize)
	vectors := make([][]float32, len(texts))

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(workers)

	for _, b := range batches {
		b := b
		g.Go(func() error {
			return e.processBatch(gCtx, b, vectors, total)
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return vectors, nil
}

// processBatch embeds a single batch and writes results into the shared vectors slice.
func (e *ProgressEmbedder) processBatch(ctx context.Context, b batch, vectors [][]float32, total int64) error {
	batchVectors, err := e.base.EmbedDocuments(ctx, b.texts)
	if err != nil {
		return err
	}
	if len(batchVectors) != len(b.indexes) {
		return fmt.Errorf("embedder returned %d vectors for batch of %d texts", len(batchVectors), len(b.indexes))
	}

	for i, vec := range batchVectors {
		idx := b.indexes[i]
		vectors[idx] = vec

		current := e.processed.Add(1)
		slog.InfoContext(ctx, "embedding chunk",
			"progress", fmt.Sprintf("%d/%d", current, total),
			"chunk_index", idx+1,
		)
	}
	return nil
}

// buildBatches splits texts into consecutive batches of at most batchSize entries.
func buildBatches(texts []string, batchSize int) []batch {
	batches := make([]batch, 0, (len(texts)+batchSize-1)/batchSize)
	for start := 0; start < len(texts); start += batchSize {
		end := start + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		indexes := make([]int, end-start)
		for i := range indexes {
			indexes[i] = start + i
		}

		batches = append(batches, batch{
			indexes: indexes,
			texts:   texts[start:end],
		})
	}
	return batches
}
