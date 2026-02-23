package cache

import (
	"testing"
	"time"
)

func TestEmbeddingGetSetAndStats(t *testing.T) {
	c := NewEmbedding(time.Minute, time.Minute)

	if got, ok := c.Get("q1"); ok || got != nil {
		t.Fatalf("expected cache miss, got %v %v", got, ok)
	}

	original := []float32{1, 2, 3}
	c.Set("q1", original)
	original[0] = 9

	got, ok := c.Get("q1")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if got[0] != 1 {
		t.Fatalf("cache should store a copy, got %v", got)
	}

	got[1] = 99
	again, ok := c.Get("q1")
	if !ok {
		t.Fatal("expected second cache hit")
	}
	if again[1] != 2 {
		t.Fatalf("Get should return copy, got %v", again)
	}

	stats := c.Stats()
	if stats.Hits != 2 || stats.Misses != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if stats.Size != 1 {
		t.Fatalf("expected size 1, got %d", stats.Size)
	}
}

func TestEmbeddingSetIgnoresEmpty(t *testing.T) {
	c := NewEmbedding(time.Minute, time.Minute)
	c.Set("q", nil)
	c.Set("q", []float32{})

	stats := c.Stats()
	if stats.Size != 0 {
		t.Fatalf("expected empty cache, got %+v", stats)
	}
}
