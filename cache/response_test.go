package cache

import (
	"testing"
	"time"

	"github.com/raykavin/gorag/pkg/models"
)

func TestResponseCacheSetGetCloneAndStats(t *testing.T) {
	c := NewResponseCache(time.Minute, time.Minute, 10)
	errText := "fail"
	resp := &models.ChatResponse{
		Output: "hello",
		ToolCalls: []models.ToolCallLog{{
			Name: "search_query",
		}},
		Error: &errText,
	}

	c.Set("Hello", "History", resp, nil, nil)

	got, ok := c.Get("hello", "history")
	if !ok || got == nil {
		t.Fatal("expected cache hit")
	}
	if got == resp {
		t.Fatal("expected cloned response")
	}

	got.Output = "changed"
	if resp.Output != "hello" {
		t.Fatalf("source response mutated: %q", resp.Output)
	}

	stats := c.Stats()
	if stats.Hits != 1 || stats.Misses != 0 || stats.Size != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestResponseCacheInvalidate(t *testing.T) {
	c := NewResponseCache(time.Minute, time.Minute, 10)
	c.Set("weather now", "", &models.ChatResponse{Output: "a"}, nil, nil)
	c.Set("sports score", "", &models.ChatResponse{Output: "b"}, nil, nil)

	c.Invalidate("WEATHER!")
	if _, ok := c.Get("weather now", ""); ok {
		t.Fatal("expected weather entry to be invalidated")
	}
	if _, ok := c.Get("sports score", ""); !ok {
		t.Fatal("expected sports entry to remain")
	}

	c.Invalidate("")
	if c.Stats().Size != 0 {
		t.Fatalf("expected flush to clear cache, stats=%+v", c.Stats())
	}
}

func TestResponseCacheTTLandEviction(t *testing.T) {
	c := NewResponseCache(time.Hour, time.Minute, 1)
	shortTools := map[string]struct{}{"weather": {}}
	mediumTools := map[string]struct{}{"news": {}}

	c.Set("first", "", &models.ChatResponse{Output: "1"}, shortTools, mediumTools)
	time.Sleep(10 * time.Millisecond)
	c.Set("second", "", &models.ChatResponse{Output: "2"}, shortTools, mediumTools)

	if c.Stats().Size != 1 {
		t.Fatalf("expected max size 1 after eviction, got %+v", c.Stats())
	}

	if ttl := c.resolveTTL(&models.ChatResponse{ToolCalls: []models.ToolCallLog{{Name: " weather "}}}, shortTools, mediumTools); ttl != 30*time.Second {
		t.Fatalf("unexpected short TTL: %v", ttl)
	}
	if ttl := c.resolveTTL(&models.ChatResponse{ToolCalls: []models.ToolCallLog{{Name: "news"}}}, shortTools, mediumTools); ttl != 2*time.Minute {
		t.Fatalf("unexpected medium TTL: %v", ttl)
	}
	if ttl := c.resolveTTL(&models.ChatResponse{}, shortTools, mediumTools); ttl != time.Hour {
		t.Fatalf("unexpected default TTL: %v", ttl)
	}
}
