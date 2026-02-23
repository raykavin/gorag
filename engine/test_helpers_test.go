package engine

import (
	"context"
	"sync"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

type mockEmbedder struct {
	mu         sync.Mutex
	embedQuery func(ctx context.Context, text string) ([]float32, error)
	embedDocs  func(ctx context.Context, texts []string) ([][]float32, error)
	queryCalls []string
	docsCalls  [][]string
}

func (m *mockEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	m.mu.Lock()
	m.queryCalls = append(m.queryCalls, text)
	fn := m.embedQuery
	m.mu.Unlock()
	if fn == nil {
		return []float32{1}, nil
	}
	return fn(ctx, text)
}

func (m *mockEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	m.mu.Lock()
	cpy := append([]string(nil), texts...)
	m.docsCalls = append(m.docsCalls, cpy)
	fn := m.embedDocs
	m.mu.Unlock()
	if fn == nil {
		out := make([][]float32, len(texts))
		for i := range texts {
			out[i] = []float32{float32(i + 1)}
		}
		return out, nil
	}
	return fn(ctx, texts)
}

type similarityCall struct {
	Query         string
	NumDocuments  int
	ScoreThreshold float32
	OptionCount   int
}

type mockVectorStore struct {
	mu       sync.Mutex
	searchFn func(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error)
	calls    []similarityCall
}

func (m *mockVectorStore) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) ([]string, error) {
	return nil, nil
}

func (m *mockVectorStore) SimilaritySearch(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
	cfg := vectorstores.Options{}
	for _, opt := range options {
		opt(&cfg)
	}

	m.mu.Lock()
	m.calls = append(m.calls, similarityCall{
		Query: query,
		NumDocuments: numDocuments,
		ScoreThreshold: cfg.ScoreThreshold,
		OptionCount: len(options),
	})
	fn := m.searchFn
	m.mu.Unlock()

	if fn == nil {
		return nil, nil
	}
	return fn(ctx, query, numDocuments, options...)
}

type mockReranker struct {
	calls int
	fn    func(ctx context.Context, query string, docs []schema.Document, topK int) ([]schema.Document, error)
}

func (m *mockReranker) Rerank(ctx context.Context, query string, docs []schema.Document, topK int) ([]schema.Document, error) {
	m.calls++
	if m.fn == nil {
		return docs, nil
	}
	return m.fn(ctx, query, docs, topK)
}
