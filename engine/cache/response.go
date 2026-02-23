package cache

import (
	"gorag/engine/models"
	"sort"
	"strings"
	"sync"
	"time"

	gocache "github.com/patrickmn/go-cache"
)

// ResponseCache stores the complete responses by normalized hash
type ResponseCache struct {
	store      *gocache.Cache
	maxItems   int
	defaultTTL time.Duration
	mu         sync.RWMutex
	hits       int64
	misses     int64
	keyIndex   map[string]string
}

// NewResponseCache creates a new instance of ResponseCache
func NewResponseCache(ttl, cleanup time.Duration, maxItems int) *ResponseCache {
	if maxItems <= 0 {
		maxItems = 1000
	}

	return &ResponseCache{
		store:      gocache.New(ttl, cleanup),
		maxItems:   maxItems,
		defaultTTL: ttl,
		keyIndex:   make(map[string]string),
	}
}

// Get retrieves the cached response for a given text and history key.
// It returns the cloned response and a boolean indicating if the key was found.
func (c *ResponseCache) Get(text, historyKey string) (*models.ChatResponse, bool) {
	key, normalized := buildResponseCacheKey(text, historyKey)
	_ = normalized

	if v, ok := c.store.Get(key); ok {
		resp, okCast := v.(*models.ChatResponse)
		if okCast {
			c.mu.Lock()
			c.hits++
			c.mu.Unlock()
			return cloneChatResponse(resp), true
		}
	}

	c.mu.Lock()
	c.misses++
	c.mu.Unlock()
	return nil, false
}

// Set stores the response for the given text and history key in the cache.
// It handles eviction if the cache size exceeds maxItems and determines TTL based on tool usage.
func (c *ResponseCache) Set(
	text string,
	historyKey string,
	resp *models.ChatResponse,
	shortTools,
	mediumTools map[string]struct{},
) {
	if resp == nil {
		return
	}

	key, normalized := buildResponseCacheKey(text, historyKey)

	c.mu.Lock()
	if c.store.ItemCount() >= c.maxItems {
		c.evictOldestLocked(1)
	}
	c.keyIndex[key] = normalized
	c.mu.Unlock()

	ttl := c.resolveTTL(resp, shortTools, mediumTools)
	c.store.Set(key, cloneChatResponse(resp), ttl)
}

// Stats returns the current metrics of the response cache.
func (c *ResponseCache) Stats() CacheStats {
	c.mu.RLock()
	hits := c.hits
	misses := c.misses
	size := c.store.ItemCount()
	c.mu.RUnlock()

	total := hits + misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}

	return CacheStats{
		Hits:    hits,
		Misses:  misses,
		Size:    size,
		HitRate: hitRate,
	}
}

// Invalidate removes items from the cache that match a specific pattern.
// If the pattern is empty, it flushes the entire cache.
func (c *ResponseCache) Invalidate(pattern string) {
	normalizedPattern := normalizeText(pattern)

	c.mu.Lock()
	defer c.mu.Unlock()

	if normalizedPattern == "" {
		c.store.Flush()
		c.keyIndex = make(map[string]string)
		return
	}

	for key, normalized := range c.keyIndex {
		if strings.Contains(normalized, normalizedPattern) {
			c.store.Delete(key)
			delete(c.keyIndex, key)
		}
	}
}

// resolveTTL determines the appropriate TTL for a response based on the tools used.
func (c *ResponseCache) resolveTTL(
	resp *models.ChatResponse,
	shortTools,
	mediumTools map[string]struct{},
) time.Duration {
	foundMedium := false
	for _, call := range resp.ToolCalls {
		name := strings.TrimSpace(strings.ToLower(call.Name))
		if _, ok := shortTools[name]; ok {
			return 30 * time.Second
		}
		if _, ok := mediumTools[name]; ok {
			foundMedium = true
		}
	}

	if foundMedium {
		return 2 * time.Minute
	}
	return c.defaultTTL
}

// evictOldestLocked removes the oldest items from the
// cache based on expiration time. This is called when
// the cache size exceeds maxItems.
func (c *ResponseCache) evictOldestLocked(target int) {
	items := c.store.Items()
	if len(items) == 0 {
		return
	}

	type cacheItem struct {
		key        string
		expiration int64
	}
	ordered := make([]cacheItem, 0, len(items))
	for key, item := range items {
		ordered = append(ordered, cacheItem{
			key:        key,
			expiration: item.Expiration,
		})
	}

	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].expiration < ordered[j].expiration
	})

	if target > len(ordered) {
		target = len(ordered)
	}

	for i := 0; i < target; i++ {
		c.store.Delete(ordered[i].key)
		delete(c.keyIndex, ordered[i].key)
	}
}
