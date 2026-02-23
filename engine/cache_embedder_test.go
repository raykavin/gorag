package engine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/raykavin/gorag/cache"
)

func TestNewCacheEmbedderValidation(t *testing.T) {
	c := cache.NewEmbedding(time.Minute, time.Minute)
	if _, err := NewCacheEmbedder(nil, c); !errors.Is(err, ErrInvalidEmbeddingBase) {
		t.Fatalf("expected ErrInvalidEmbeddingBase, got %v", err)
	}
	if _, err := NewCacheEmbedder(&mockEmbedder{}, nil); !errors.Is(err, ErrInvalidCache) {
		t.Fatalf("expected ErrInvalidCache, got %v", err)
	}
}

func TestCacheEmbedderEmbedQueryUsesCache(t *testing.T) {
	base := &mockEmbedder{}
	c := cache.NewEmbedding(time.Minute, time.Minute)
	e, err := NewCacheEmbedder(base, c)
	if err != nil {
		t.Fatal(err)
	}

	first, err := e.EmbedQuery(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	second, err := e.EmbedQuery(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(base.queryCalls) != 1 {
		t.Fatalf("expected 1 base EmbedQuery call, got %d", len(base.queryCalls))
	}
	if first[0] != second[0] {
		t.Fatalf("unexpected vectors %v %v", first, second)
	}
}

func TestCacheEmbedderEmbedDocumentsMixedCache(t *testing.T) {
	base := &mockEmbedder{
		embedDocs: func(ctx context.Context, texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{float32(len(texts[i]))}
			}
			return out, nil
		},
	}
	c := cache.NewEmbedding(time.Minute, time.Minute)
	c.Set("b", []float32{99})

	e, err := NewCacheEmbedder(base, c)
	if err != nil {
		t.Fatal(err)
	}

	vectors, err := e.EmbedDocuments(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(vectors))
	}
	if vectors[1][0] != 99 {
		t.Fatalf("expected cached vector for b, got %v", vectors[1])
	}
	if len(base.docsCalls) != 1 {
		t.Fatalf("expected one base docs call, got %d", len(base.docsCalls))
	}
	if got := base.docsCalls[0]; len(got) != 2 || got[0] != "a" || got[1] != "c" {
		t.Fatalf("unexpected missing docs set: %v", got)
	}

	if got, ok := c.Get("a"); !ok || got[0] != 1 {
		t.Fatalf("expected cached embedding for a, got %v %v", got, ok)
	}
	if got, ok := c.Get("c"); !ok || got[0] != 1 {
		t.Fatalf("expected cached embedding for c, got %v %v", got, ok)
	}
}

func TestCacheEmbedderEmbedDocumentsInvalidQuantity(t *testing.T) {
	base := &mockEmbedder{
		embedDocs: func(ctx context.Context, texts []string) ([][]float32, error) {
			return [][]float32{{1}}, nil
		},
	}
	c := cache.NewEmbedding(time.Minute, time.Minute)
	e, _ := NewCacheEmbedder(base, c)

	_, err := e.EmbedDocuments(context.Background(), []string{"a", "b"})
	if !errors.Is(err, ErrInvalidReturnableVectorQuantity) {
		t.Fatalf("expected ErrInvalidReturnableVectorQuantity, got %v", err)
	}
}
