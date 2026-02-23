package engine

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestNewProgressEmbedderValidation(t *testing.T) {
	if _, err := NewProgressEmbedder(nil, 1, 1); !errors.Is(err, ErrNilEmbedder) {
		t.Fatalf("expected ErrNilEmbedder, got %v", err)
	}

	m := &mockEmbedder{}
	e, err := NewProgressEmbedder(m, -1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if e.total != 0 || e.workers != defEmbedWorkers || e.batchSize != defEmbedBatchSize {
		t.Fatalf("unexpected defaults: %+v", e)
	}
}

func TestProgressEmbedderEmbedDocumentsPreservesOrder(t *testing.T) {
	m := &mockEmbedder{
		embedDocs: func(ctx context.Context, texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i, txt := range texts {
				out[i] = []float32{float32(len(txt))}
			}
			return out, nil
		},
	}
	e, _ := NewProgressEmbedder(m, 0, 3)
	e.batchSize = 2

	got, err := e.EmbedDocuments(context.Background(), []string{"a", "bbbb", "cc", "ddd"})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{1}, {4}, {2}, {3}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("vectors mismatch: got %v want %v", got, want)
	}
}

func TestProgressEmbedderEmbedDocumentsErrors(t *testing.T) {
	e := &ProgressEmbedder{}
	if _, err := e.EmbedDocuments(context.Background(), []string{"a"}); !errors.Is(err, ErrUninitializedEmbedder) {
		t.Fatalf("expected uninitialized error, got %v", err)
	}

	m := &mockEmbedder{
		embedDocs: func(ctx context.Context, texts []string) ([][]float32, error) {
			return [][]float32{{1}}, nil
		},
	}
	e2, _ := NewProgressEmbedder(m, 0, 1)
	e2.batchSize = 2
	_, err := e2.EmbedDocuments(context.Background(), []string{"a", "b"})
	if err == nil || err.Error() != "embedder returned 1 vectors for batch of 2 texts" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestBuildBatches(t *testing.T) {
	batches := buildBatches([]string{"a", "b", "c", "d", "e"}, 2)
	if len(batches) != 3 {
		t.Fatalf("expected 3 batches, got %d", len(batches))
	}
	if fmt.Sprint(batches[0].indexes) != "[0 1]" || fmt.Sprint(batches[2].indexes) != "[4]" {
		t.Fatalf("unexpected indexes: %+v", batches)
	}
}
