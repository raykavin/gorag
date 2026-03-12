package engine

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"

	"github.com/raykavin/gorag/cache"
)

// RetrievalMode defines the document retrieval strategy.
type RetrievalMode string

const (
	RetrievalDense          RetrievalMode = "dense"
	RetrievalReranker       RetrievalMode = "reranker"
	RetrievalHybrid         RetrievalMode = "hybrid"
	RetrievalHybridReranker RetrievalMode = "hybrid_reranker"

	defaultTopK              = 3
	defaultMinScore          = float32(0.7)
	defaultContextMaxChars   = 6000
	defaultChunkMaxChars     = 1200
	defaultHybridMultiplier  = 3
	defaultRRFConstant       = 60.0
	bm25K1                   = 1.2
	bm25B                    = 0.75
	defaultMinTokenLength    = 2
	defaultTruncateLookback  = 48
	defaultTruncateEllipsis  = "..."
	defaultChunkIDPrefix     = "rank_"
	defaultContextHeaderOpen = "["
	defaultContextHeaderEnd  = "]"
)

// RAGEngine retrieves relevant document chunks from a vector store
// and optionally reranks them before injecting into a prompt.
type RAGEngine struct {
	store         vectorstores.VectorStore
	reranker      DocumentReranker
	embCache      *cache.Embedding
	keywordIndex  *keywordIndex
	systemPrompt  string
	retrievalMode RetrievalMode
	topK          int
	minScore      float32
	rerankerK     int
	chunksLoaded  int
	contextMax    int
	chunkMax      int
	forceAlways   bool
}

// Config holds the configuration parameters for a RAGEngine.
type Config struct {
	TopK          int
	MinScore      float32
	ForceAlways   bool
	RetrievalMode string
	RerankerK     int
	ChunksLoaded  int
	KeywordDocs   []schema.Document
	ContextMax    int
	ChunkMax      int
	SystemPrompt  string
}

// NewRAGEngine creates a RAGEngine with the given store, reranker, cache and config.
func NewRAGEngine(
	store vectorstores.VectorStore,
	reranker DocumentReranker,
	embCache *cache.Embedding,
	cfg Config,
) *RAGEngine {
	if cfg.TopK <= 0 {
		cfg.TopK = defaultTopK
	}
	if cfg.MinScore <= 0 {
		cfg.MinScore = defaultMinScore
	}
	if cfg.RerankerK < cfg.TopK {
		cfg.RerankerK = cfg.TopK
	}
	if cfg.ContextMax <= 0 {
		cfg.ContextMax = defaultContextMaxChars
	}
	if cfg.ChunkMax <= 0 {
		cfg.ChunkMax = defaultChunkMaxChars
	}
	if cfg.ChunkMax > cfg.ContextMax {
		cfg.ChunkMax = cfg.ContextMax
	}

	return &RAGEngine{
		store:         store,
		reranker:      reranker,
		embCache:      embCache,
		topK:          cfg.TopK,
		minScore:      cfg.MinScore,
		forceAlways:   cfg.ForceAlways,
		rerankerK:     cfg.RerankerK,
		chunksLoaded:  cfg.ChunksLoaded,
		keywordIndex:  newKeywordIndex(cfg.KeywordDocs),
		contextMax:    cfg.ContextMax,
		chunkMax:      cfg.ChunkMax,
		systemPrompt:  strings.TrimSpace(cfg.SystemPrompt),
		retrievalMode: normalizeRetrievalMode(cfg.RetrievalMode),
	}
}

// Query retrieves relevant context for the given query and returns it
// formatted as a prompt block. The boolean return indicates whether any
// context was found.
func (r *RAGEngine) Query(ctx context.Context, query string) (string, bool, error) {
	if r == nil {
		return "", false, nil
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return "", false, nil
	}

	results, err := r.retrieve(ctx, query)
	if err != nil {
		return "", false, err
	}

	results = filterByMinScore(results, r.minScore)
	if len(results) == 0 {
		return "", false, nil
	}

	context := buildContextBlock(results, r.contextMax, r.chunkMax)
	if context == "" {
		return "", false, nil
	}

	return context, true, nil
}

// ChunksLoaded returns the number of knowledge chunks loaded into the store.
func (r *RAGEngine) ChunksLoaded() int {
	if r == nil {
		return 0
	}
	return r.chunksLoaded
}

// EmbeddingCacheStats returns cache statistics for the embedding cache.
func (r *RAGEngine) EmbeddingCacheStats() cache.CacheStats {
	if r == nil || r.embCache == nil {
		return cache.CacheStats{}
	}
	return r.embCache.Stats()
}

// SystemPrompt returns the fixed system prompt context loaded at bootstrap.
func (r *RAGEngine) SystemPrompt() string {
	if r == nil {
		return ""
	}
	return r.systemPrompt
}

// retrieve dispatches to the appropriate retrieval strategy.
func (r *RAGEngine) retrieve(ctx context.Context, query string) ([]schema.Document, error) {
	switch r.retrievalMode {
	case RetrievalReranker:
		return r.retrieveWithReranker(ctx, query)
	case RetrievalHybrid:
		return r.retrieveHybrid(ctx, query, false)
	case RetrievalHybridReranker:
		return r.retrieveHybrid(ctx, query, true)
	default:
		return r.searchDense(ctx, query, r.topK)
	}
}

// retrieveWithReranker performs dense retrieval followed by reranking.
func (r *RAGEngine) retrieveWithReranker(ctx context.Context, query string) ([]schema.Document, error) {
	if r.reranker == nil {
		return nil, fmt.Errorf("reranker mode is active but no reranker was provided")
	}

	candidates, err := r.searchDenseCandidates(ctx, query)
	if err != nil || len(candidates) == 0 {
		return nil, err
	}

	reranked, err := r.reranker.Rerank(ctx, query, candidates, r.topK)
	if err != nil {
		return nil, fmt.Errorf("reranker failed: %w", err)
	}
	return reranked, nil
}

// retrieveHybrid fuses dense and keyword retrieval with RRF and optionally reranks.
func (r *RAGEngine) retrieveHybrid(ctx context.Context, query string, withReranker bool) ([]schema.Document, error) {
	if withReranker && r.reranker == nil {
		return nil, fmt.Errorf("hybrid_reranker mode is active but no reranker was provided")
	}

	candidatesK := r.hybridCandidatesK()

	dense, err := r.searchDense(ctx, query, candidatesK)
	if err != nil {
		return nil, err
	}
	dense = filterByMinScore(dense, r.minScore)
	if len(dense) == 0 {
		return nil, nil
	}

	denseKeys := docKeySet(dense)
	keyword := filterDocsByKeys(r.searchKeyword(query, candidatesK), denseKeys)
	candidates := reciprocalRankFusion(candidatesK, dense, keyword)
	if len(candidates) == 0 {
		return nil, nil
	}

	// Cap candidates before optional reranking, then trim to topK.
	limit := r.topK
	if withReranker {
		limit = r.rerankerK
	}
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}

	if withReranker {
		reranked, err := r.reranker.Rerank(ctx, query, candidates, r.topK)
		if err != nil {
			return nil, fmt.Errorf("reranker failed: %w", err)
		}
		return reranked, nil
	}
	return candidates, nil
}

// searchDense performs a similarity search with an optional score threshold.
// When forceAlways is true and no results pass the threshold, it retries without one.
func (r *RAGEngine) searchDense(ctx context.Context, query string, topK int) ([]schema.Document, error) {
	if r.store == nil {
		return nil, nil
	}
	if topK <= 0 {
		topK = r.topK
	}

	results, err := r.store.SimilaritySearch(ctx, query, topK, vectorstores.WithScoreThreshold(r.minScore))
	if err != nil {
		return nil, fmt.Errorf("semantic search failed: %w", err)
	}

	if len(results) == 0 && r.forceAlways {
		results, err = r.store.SimilaritySearch(ctx, query, topK)
		if err != nil {
			return nil, fmt.Errorf("semantic search fallback failed: %w", err)
		}
	}
	return results, nil
}

// searchDenseCandidates retrieves rerankerK candidates, falling back to a no-threshold
// search to improve recall when the scored search returns too few results.
func (r *RAGEngine) searchDenseCandidates(ctx context.Context, query string) ([]schema.Document, error) {
	candidates, err := r.searchDense(ctx, query, r.rerankerK)
	if err != nil {
		return nil, err
	}
	if len(candidates) >= r.rerankerK || r.store == nil {
		return candidates, nil
	}

	fallback, err := r.store.SimilaritySearch(ctx, query, r.rerankerK)
	if err != nil {
		return nil, fmt.Errorf("candidate search fallback failed: %w", err)
	}
	return mergeUniqueDocuments(candidates, fallback, r.rerankerK), nil
}

func (r *RAGEngine) hybridCandidatesK() int {
	k := r.topK * defaultHybridMultiplier
	if k < r.rerankerK {
		k = r.rerankerK
	}
	if k < r.topK {
		k = r.topK
	}
	return k
}

// searchKeyword performs lexical retrieval over the in-memory keyword index.
func (r *RAGEngine) searchKeyword(query string, topK int) []schema.Document {
	if r.keywordIndex == nil {
		return nil
	}
	if topK <= 0 {
		topK = r.topK
	}
	return r.keywordIndex.search(query, topK)
}

// buildContextBlock formats retrieved docs into a prompt-ready context block.
func buildContextBlock(docs []schema.Document, maxContextChars, maxChunkChars int) string {
	if maxContextChars <= 0 {
		maxContextChars = defaultContextMaxChars
	}
	if maxChunkChars <= 0 {
		maxChunkChars = defaultChunkMaxChars
	}
	if maxChunkChars > maxContextChars {
		maxChunkChars = maxContextChars
	}

	var b strings.Builder

	for idx, doc := range docs {
		content := strings.TrimSpace(doc.PageContent)
		if content == "" {
			continue
		}

		content = truncateText(content, maxChunkChars)
		header := contextHeader(idx+1, doc)
		entry := header + "\n" + content

		prefix := ""
		if b.Len() > 0 {
			prefix = "\n\n"
		}

		if b.Len()+len(prefix)+len(entry) > maxContextChars {
			if b.Len() > 0 {
				break
			}
			allowed := maxContextChars - len(header) - 1
			if allowed <= 0 {
				break
			}
			entry = header + "\n" + truncateText(content, allowed)
		}

		if prefix != "" {
			_, _ = b.WriteString(prefix)
		}
		_, _ = b.WriteString(entry)
	}

	return strings.TrimSpace(b.String())
}

func contextHeader(rank int, doc schema.Document) string {
	id, ok := metadataString(doc.Metadata, "id")
	if !ok {
		id = defaultChunkIDPrefix + fmt.Sprintf("%d", rank)
	}

	parts := []string{"chunk_id=" + id}
	if source, ok := metadataString(doc.Metadata, "source"); ok {
		parts = append(parts, "source="+source)
	}
	if category, ok := metadataString(doc.Metadata, "category"); ok {
		parts = append(parts, "category="+category)
	}
	if doc.Score > 0 {
		parts = append(parts, fmt.Sprintf("score=%.4f", doc.Score))
	}

	return defaultContextHeaderOpen + strings.Join(parts, " | ") + defaultContextHeaderEnd
}

// reciprocalRankFusion merges ranked lists using RRF scoring.
func reciprocalRankFusion(limit int, rankedLists ...[]schema.Document) []schema.Document {
	if limit <= 0 {
		limit = defaultTopK
	}

	type scoredDoc struct {
		doc   schema.Document
		score float64
		key   string
	}

	byKey := make(map[string]*scoredDoc, limit*2)
	for _, list := range rankedLists {
		for rank, doc := range list {
			if strings.TrimSpace(doc.PageContent) == "" {
				continue
			}
			key := documentKey(doc)
			score := 1.0 / (defaultRRFConstant + float64(rank+1))

			current, ok := byKey[key]
			if !ok {
				copied := doc
				byKey[key] = &scoredDoc{doc: copied, score: score, key: key}
				continue
			}
			current.score += score
			// Preserve the best native score (e.g., vector similarity) so that
			// MinScore semantics remain stable after fusion.
			if doc.Score > current.doc.Score {
				current.doc.Score = doc.Score
			}
		}
	}

	if len(byKey) == 0 {
		return nil
	}

	all := make([]scoredDoc, 0, len(byKey))
	for _, item := range byKey {
		all = append(all, *item)
	}

	sort.SliceStable(all, func(i, j int) bool {
		if all[i].score == all[j].score {
			return all[i].key < all[j].key
		}
		return all[i].score > all[j].score
	})

	if len(all) > limit {
		all = all[:limit]
	}

	out := make([]schema.Document, 0, len(all))
	for _, item := range all {
		doc := item.doc
		if doc.Metadata == nil {
			doc.Metadata = map[string]any{}
		}
		doc.Metadata["hybrid_rrf_score"] = item.score
		out = append(out, doc)
	}
	return out
}

// filterByMinScore returns a new slice containing only documents whose score
// meets or exceeds minScore. The original slice is never modified.
func filterByMinScore(docs []schema.Document, minScore float32) []schema.Document {
	if minScore <= 0 || len(docs) == 0 {
		return docs
	}
	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if doc.Score >= minScore {
			filtered = append(filtered, doc)
		}
	}
	return filtered
}

// filterDocsByKeys returns a new slice containing only documents whose key
// appears in allowed. The original slice is never modified.
func filterDocsByKeys(docs []schema.Document, allowed map[string]struct{}) []schema.Document {
	if len(docs) == 0 || len(allowed) == 0 {
		return nil
	}
	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if _, ok := allowed[documentKey(doc)]; ok {
			filtered = append(filtered, doc)
		}
	}
	return filtered
}

// docKeySet builds a set of document keys from a slice, used for O(1) lookups.
func docKeySet(docs []schema.Document) map[string]struct{} {
	keys := make(map[string]struct{}, len(docs))
	for _, doc := range docs {
		keys[documentKey(doc)] = struct{}{}
	}
	return keys
}

// mergeUniqueDocuments merges two document slices, deduplicating by key, up to topK results.
func mergeUniqueDocuments(first, second []schema.Document, topK int) []schema.Document {
	capacity := len(first) + len(second)
	if topK > 0 && topK < capacity {
		capacity = topK
	}

	out := make([]schema.Document, 0, capacity)
	seen := make(map[string]struct{}, len(first)+len(second))

	for _, batch := range [][]schema.Document{first, second} {
		for _, doc := range batch {
			if topK > 0 && len(out) >= topK {
				return out
			}
			if strings.TrimSpace(doc.PageContent) == "" {
				continue
			}
			key := documentKey(doc)
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, doc)
		}
	}
	return out
}

// documentKey returns a stable deduplication key for a document,
// preferring metadata "id", then "source", then falling back to content.
func documentKey(doc schema.Document) string {
	if doc.Metadata != nil {
		if id, ok := metadataString(doc.Metadata, "id"); ok {
			return "id:" + id
		}
		if source, ok := metadataString(doc.Metadata, "source"); ok {
			return "source:" + source
		}
	}
	return "content:" + strings.TrimSpace(doc.PageContent)
}

// metadataString extracts a non-empty string value from a metadata map.
func metadataString(meta map[string]any, key string) (string, bool) {
	raw, ok := meta[key]
	if !ok {
		return "", false
	}
	s, ok := raw.(string)
	if !ok {
		return "", false
	}
	s = strings.TrimSpace(s)
	return s, s != ""
}

// normalizeRetrievalMode maps a raw string to a RetrievalMode, defaulting to dense.
func normalizeRetrievalMode(mode string) RetrievalMode {
	switch RetrievalMode(strings.ToLower(strings.TrimSpace(mode))) {
	case RetrievalReranker:
		return RetrievalReranker
	case RetrievalHybrid:
		return RetrievalHybrid
	case RetrievalHybridReranker, "hybrid-reranker":
		return RetrievalHybridReranker
	default:
		return RetrievalDense
	}
}

type posting struct {
	docIndex int
	tf       int
}

type keywordIndex struct {
	docs      []schema.Document
	postings  map[string][]posting
	docLen    []int
	docFreq   map[string]int
	avgDocLen float64
}

func newKeywordIndex(docs []schema.Document) *keywordIndex {
	if len(docs) == 0 {
		return nil
	}

	idx := &keywordIndex{
		docs:     make([]schema.Document, 0, len(docs)),
		postings: make(map[string][]posting, len(docs)*4),
		docLen:   make([]int, 0, len(docs)),
		docFreq:  make(map[string]int, len(docs)*4),
	}

	totalLen := 0
	for _, doc := range docs {
		content := strings.TrimSpace(doc.PageContent)
		if content == "" {
			continue
		}

		tokens := tokenize(content)
		if len(tokens) == 0 {
			continue
		}

		docIndex := len(idx.docs)
		idx.docs = append(idx.docs, doc)
		idx.docLen = append(idx.docLen, len(tokens))
		totalLen += len(tokens)

		tf := make(map[string]int, len(tokens))
		for _, token := range tokens {
			tf[token]++
		}
		for term, count := range tf {
			idx.postings[term] = append(
				idx.postings[term],
				posting{docIndex: docIndex, tf: count},
			)
			idx.docFreq[term]++
		}
	}

	if len(idx.docs) == 0 {
		return nil
	}
	idx.avgDocLen = float64(totalLen) / float64(len(idx.docs))
	return idx
}

func (k *keywordIndex) search(query string, limit int) []schema.Document {
	if k == nil || len(k.docs) == 0 {
		return nil
	}
	if limit <= 0 {
		limit = defaultTopK
	}

	queryTokens := tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}

	seenTerms := make(map[string]struct{}, len(queryTokens))
	docScores := make(map[int]float64, limit*4)
	n := float64(len(k.docs))

	for _, term := range queryTokens {
		if _, done := seenTerms[term]; done {
			continue
		}
		seenTerms[term] = struct{}{}

		postings, ok := k.postings[term]
		if !ok {
			continue
		}

		df := float64(k.docFreq[term])
		idf := math.Log(1 + (n-df+0.5)/(df+0.5))

		for _, p := range postings {
			dl := float64(k.docLen[p.docIndex])
			tf := float64(p.tf)
			denom := tf + bm25K1*(1-bm25B+bm25B*(dl/k.avgDocLen))
			if denom == 0 {
				continue
			}
			docScores[p.docIndex] += idf * ((tf * (bm25K1 + 1)) / denom)
		}
	}

	if len(docScores) == 0 {
		return nil
	}

	type scored struct {
		index int
		score float64
	}

	scoredDocs := make([]scored, 0, len(docScores))
	for idx, score := range docScores {
		scoredDocs = append(scoredDocs, scored{index: idx, score: score})
	}

	sort.SliceStable(scoredDocs, func(i, j int) bool {
		if scoredDocs[i].score == scoredDocs[j].score {
			return scoredDocs[i].index < scoredDocs[j].index
		}
		return scoredDocs[i].score > scoredDocs[j].score
	})

	if len(scoredDocs) > limit {
		scoredDocs = scoredDocs[:limit]
	}

	out := make([]schema.Document, 0, len(scoredDocs))
	for _, item := range scoredDocs {
		doc := k.docs[item.index]
		doc.Score = float32(item.score)
		out = append(out, doc)
	}
	return out
}

func tokenize(text string) []string {
	text = strings.TrimSpace(strings.ToLower(text))
	if text == "" {
		return nil
	}

	var tokens []string
	var b strings.Builder

	flush := func() {
		if b.Len() == 0 {
			return
		}
		token := b.String()
		b.Reset()
		if len([]rune(token)) < defaultMinTokenLength {
			return
		}
		tokens = append(tokens, token)
	}

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			_, _ = b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()

	return tokens
}

// truncateText shortens text to at most maxChars runes, breaking on a word
// boundary within the last defaultTruncateLookback runes when possible,
// and appending an ellipsis.
func truncateText(text string, maxChars int) string {
	text = strings.TrimSpace(text)
	if text == "" || maxChars <= 0 {
		return ""
	}

	runes := []rune(text)
	if len(runes) <= maxChars {
		return text
	}

	end := maxChars
	lookback := defaultTruncateLookback
	if lookback > end {
		lookback = end
	}

	// Walk back from the cut point to find a word boundary.
	for i := end - 1; i >= end-lookback; i-- {
		if unicode.IsSpace(runes[i]) {
			end = i
			break
		}
	}

	return strings.TrimSpace(string(runes[:end])) + defaultTruncateEllipsis
}
