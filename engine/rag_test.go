package engine

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/raykavin/gorag/cache"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

func TestNewRAGEngineDefaultsAndMode(t *testing.T) {
	r := NewRAGEngine(nil, nil, nil, Config{})
	if r.topK != defaultTopK || r.minScore != defaultMinScore || r.retrievalMode != RetrievalDense || r.rerankerK != defaultTopK {
		t.Fatalf("unexpected defaults: %+v", r)
	}

	r2 := NewRAGEngine(nil, nil, nil, Config{TopK: 2, RerankerK: 1, RetrievalMode: "RERANKER"})
	if r2.rerankerK != 2 || r2.retrievalMode != RetrievalReranker {
		t.Fatalf("unexpected config normalization: %+v", r2)
	}
}

func TestRAGEngineQueryDenseForceAlwaysFallback(t *testing.T) {
	call := 0
	store := &mockVectorStore{
		searchFn: func(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
			call++
			if call == 1 {
				return nil, nil
			}
			return []schema.Document{
				{PageContent: "first", Score: 0.91},
				{PageContent: "second", Score: 0.89},
			}, nil
		},
	}

	r := NewRAGEngine(store, nil, nil, Config{TopK: 2, ForceAlways: true})
	ctxBlock, used, err := r.Query(context.Background(), "question")
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("expected context to be used")
	}
	if !strings.Contains(ctxBlock, "chunk_id=rank_1") || !strings.Contains(ctxBlock, "first") {
		t.Fatalf("unexpected context block: %q", ctxBlock)
	}
	if len(store.calls) != 2 {
		t.Fatalf("expected 2 similarity calls (threshold + fallback), got %d", len(store.calls))
	}
	if store.calls[0].OptionCount == 0 || store.calls[1].OptionCount != 0 {
		t.Fatalf("expected options on first call only, calls=%+v", store.calls)
	}

	if got, used, err := r.Query(context.Background(), "   "); got != "" || used || err != nil {
		t.Fatalf("empty query should return no context: %q %v %v", got, used, err)
	}
}

func TestRAGEngineRerankerMode(t *testing.T) {
	store := &mockVectorStore{
		searchFn: func(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
			if len(options) == 0 {
				return []schema.Document{
					{PageContent: "doc1", Score: 0.93},
					{PageContent: "doc2", Score: 0.85},
					{PageContent: "doc3", Score: 0.80},
				}, nil
			}
			return []schema.Document{{PageContent: "doc1", Score: 0.93}}, nil
		},
	}

	r := NewRAGEngine(store, nil, nil, Config{RetrievalMode: "reranker"})
	_, _, err := r.Query(context.Background(), "q")
	if err == nil || !strings.Contains(err.Error(), "no reranker") {
		t.Fatalf("expected missing reranker error, got %v", err)
	}

	m := &mockReranker{fn: func(ctx context.Context, query string, docs []schema.Document, topK int) ([]schema.Document, error) {
		if topK != 2 {
			t.Fatalf("unexpected topK: %d", topK)
		}
		return []schema.Document{docs[0]}, nil
	}}

	r2 := NewRAGEngine(store, m, nil, Config{TopK: 2, RerankerK: 3, RetrievalMode: "reranker"})
	out, used, err := r2.Query(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if !used || !strings.Contains(out, "doc1") {
		t.Fatalf("unexpected rerank output: %q used=%v", out, used)
	}
	if m.calls != 1 {
		t.Fatalf("expected reranker call, got %d", m.calls)
	}
}

func TestRAGEngineHelpers(t *testing.T) {
	if (*RAGEngine)(nil).ChunksLoaded() != 0 {
		t.Fatal("nil RAGEngine should report zero chunks")
	}
	if (*RAGEngine)(nil).EmbeddingCacheStats().Size != 0 {
		t.Fatal("nil RAGEngine should report empty cache stats")
	}

	embCache := cache.NewEmbedding(time.Minute, time.Minute)
	embCache.Set("x", []float32{1})
	r := NewRAGEngine(nil, nil, embCache, Config{ChunksLoaded: 4})
	if r.ChunksLoaded() != 4 {
		t.Fatalf("unexpected chunks loaded: %d", r.ChunksLoaded())
	}
	if r.EmbeddingCacheStats().Size != 1 {
		t.Fatalf("unexpected embedding stats: %+v", r.EmbeddingCacheStats())
	}

	docs := mergeUniqueDocuments(
		[]schema.Document{{PageContent: "a", Metadata: map[string]any{"id": "1"}}},
		[]schema.Document{{PageContent: "dup", Metadata: map[string]any{"id": "1"}}, {PageContent: "b", Metadata: map[string]any{"source": "s"}}},
		2,
	)
	if len(docs) != 2 {
		t.Fatalf("expected deduped docs, got %d", len(docs))
	}
	if documentKey(schema.Document{PageContent: " x "}) != "content:x" {
		t.Fatalf("unexpected fallback key")
	}
	if mode := normalizeRetrievalMode("unknown"); mode != RetrievalDense {
		t.Fatalf("unexpected default mode: %v", mode)
	}
	if mode := normalizeRetrievalMode("hybrid"); mode != RetrievalHybrid {
		t.Fatalf("unexpected hybrid mode: %v", mode)
	}
	if mode := normalizeRetrievalMode("hybrid-reranker"); mode != RetrievalHybridReranker {
		t.Fatalf("unexpected hybrid reranker mode: %v", mode)
	}
}

func TestSearchDenseError(t *testing.T) {
	store := &mockVectorStore{
		searchFn: func(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
			return nil, errors.New("boom")
		},
	}
	r := NewRAGEngine(store, nil, nil, Config{})
	if _, err := r.searchDense(context.Background(), "q", 1); err == nil || !strings.Contains(err.Error(), "semantic search failed") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRAGEngineHybridRetrieval(t *testing.T) {
	store := &mockVectorStore{
		searchFn: func(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
			return []schema.Document{
				{
					PageContent: "fatura boleto segunda via",
					Metadata:    map[string]any{"id": "D1", "source": "dense"},
					Score:       0.88,
				},
				{
					PageContent: "internet lenta suporte tecnico",
					Metadata:    map[string]any{"id": "D2", "source": "dense"},
					Score:       0.76,
				},
			}, nil
		},
	}

	keywordDocs := []schema.Document{
		{PageContent: "segunda via de boleto e fatura", Metadata: map[string]any{"id": "D1", "source": "lexical"}, Score: 1.12},
		{PageContent: "apps e streaming", Metadata: map[string]any{"id": "L2", "source": "lexical"}},
	}

	r := NewRAGEngine(store, nil, nil, Config{
		TopK:          2,
		RetrievalMode: "hybrid",
		KeywordDocs:   keywordDocs,
	})

	out, used, err := r.Query(context.Background(), "preciso da segunda via do boleto")
	if err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Fatal("expected hybrid retrieval to use context")
	}
	if !strings.Contains(out, "chunk_id=D1") {
		t.Fatalf("expected hybrid context in output, got: %q", out)
	}
	if strings.Contains(out, "score=0.0164") {
		t.Fatalf("unexpected RRF score leaking as document score: %q", out)
	}
	if len(store.calls) == 0 {
		t.Fatal("expected dense retrieval call in hybrid mode")
	}
}

func TestBuildContextBlockRespectsCharBudget(t *testing.T) {
	docs := []schema.Document{
		{
			PageContent: strings.Repeat("a", 200),
			Metadata:    map[string]any{"id": "A1", "source": "s1"},
		},
		{
			PageContent: strings.Repeat("b", 200),
			Metadata:    map[string]any{"id": "B1", "source": "s2"},
		},
	}

	out := buildContextBlock(docs, 140, 80)
	if len(out) == 0 {
		t.Fatal("expected non-empty context block")
	}
	if len(out) > 140 {
		t.Fatalf("expected output <= 140 chars, got %d", len(out))
	}
	if !strings.Contains(out, "chunk_id=A1") {
		t.Fatalf("expected first chunk header in output, got %q", out)
	}
}
