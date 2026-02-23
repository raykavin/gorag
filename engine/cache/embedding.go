package cache

import (
	"sync"
	"time"

	gocache "github.com/patrickmn/go-cache"
)

// Embedding prevents the recomputation for repeated texts
type Embedding struct {
	store  *gocache.Cache
	mu     sync.RWMutex
	hits   int64
	misses int64
}

// NewEmbedding creates a new instance of Embedding
func NewEmbedding(ttl, cleanup time.Duration) *Embedding {
	return &Embedding{
		store: gocache.New(ttl, cleanup),
	}
}

// Get retrieves the embedding for the given text from the cache.
// It returns the embedding and a boolean indicating if the key was found.
func (c *Embedding) Get(text string) ([]float32, bool) {
	key := embeddingKey(text)
	if v, ok := c.store.Get(key); ok {
		if emb, okCast := v.([]float32); okCast {
			c.mu.Lock()
			c.hits++
			c.mu.Unlock()
			return append([]float32(nil), emb...), true
		}
	}

	c.mu.Lock()
	c.misses++
	c.mu.Unlock()
	return nil, false
}

// Set stores the embedding for the given text in the cache.
func (c *Embedding) Set(text string, embedding []float32) {
	if len(embedding) == 0 {
		return
	}
	key := embeddingKey(text)
	copyEmb := append([]float32(nil), embedding...)
	c.store.SetDefault(key, copyEmb)
}

func (c *Embedding) Stats() CacheStats {
	c.mu.RLock()
	hits := c.hits
	misses := c.misses
	size := c.store.ItemCount()
	c.mu.RUnlock()

	total := hits + misses
	rate := 0.0
	if total > 0 {
		rate = float64(hits) / float64(total)
	}

	return CacheStats{
		Hits:    hits,
		Misses:  misses,
		Size:    size,
		HitRate: rate,
	}
}
