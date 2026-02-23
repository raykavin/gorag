package main

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/raykavin/gorag/engine"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

// inMemoryStore is a tiny demo vector store used only for examples.
type inMemoryStore struct {
	docs []schema.Document
}

func (s *inMemoryStore) AddDocuments(
	_ context.Context,
	docs []schema.Document,
	_ ...vectorstores.Option,
) ([]string, error) {
	start := len(s.docs)
	s.docs = append(s.docs, docs...)

	ids := make([]string, len(docs))
	for i := range docs {
		ids[i] = fmt.Sprintf("doc-%d", start+i+1)
	}
	return ids, nil
}

func (s *inMemoryStore) SimilaritySearch(
	_ context.Context,
	query string,
	numDocuments int,
	_ ...vectorstores.Option,
) ([]schema.Document, error) {
	queryTokens := tokenize(query)
	if numDocuments <= 0 {
		numDocuments = 3
	}

	type scored struct {
		doc   schema.Document
		score int
	}
	scoredDocs := make([]scored, 0, len(s.docs))
	for _, doc := range s.docs {
		contentTokens := tokenize(doc.PageContent)
		score := overlapCount(queryTokens, contentTokens)
		scoredDocs = append(scoredDocs, scored{doc: doc, score: score})
	}

	sort.SliceStable(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].score > scoredDocs[j].score
	})

	out := make([]schema.Document, 0, numDocuments)
	for _, item := range scoredDocs {
		if len(out) >= numDocuments {
			break
		}
		out = append(out, item.doc)
	}
	return out, nil
}

type keywordReranker struct{}

func (keywordReranker) Rerank(
	_ context.Context,
	query string,
	docs []schema.Document,
	topK int,
) ([]schema.Document, error) {
	queryTokens := tokenize(query)
	type scored struct {
		doc   schema.Document
		score int
	}

	scoredDocs := make([]scored, 0, len(docs))
	for _, doc := range docs {
		scoredDocs = append(scoredDocs, scored{
			doc:   doc,
			score: overlapCount(queryTokens, tokenize(doc.PageContent)),
		})
	}

	sort.SliceStable(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].score > scoredDocs[j].score
	})

	if topK <= 0 || topK > len(scoredDocs) {
		topK = len(scoredDocs)
	}

	out := make([]schema.Document, 0, topK)
	for i := 0; i < topK; i++ {
		out = append(out, scoredDocs[i].doc)
	}
	return out, nil
}

func tokenize(text string) []string {
	clean := strings.ToLower(strings.TrimSpace(text))
	if clean == "" {
		return nil
	}
	fields := strings.Fields(clean)
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		token := strings.Trim(f, ".,;:!?()[]{}\"'")
		if token != "" {
			out = append(out, token)
		}
	}
	return out
}

func overlapCount(a, b []string) int {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	set := make(map[string]struct{}, len(a))
	for _, token := range a {
		set[token] = struct{}{}
	}
	count := 0
	for _, token := range b {
		if _, ok := set[token]; ok {
			count++
		}
	}
	return count
}

func main() {
	ctx := context.Background()

	store := &inMemoryStore{
		docs: []schema.Document{
			{PageContent: "Refunds are allowed within 30 days with receipt."},
			{PageContent: "Shipping usually takes 3 to 5 business days."},
			{PageContent: "Support is available Monday through Friday."},
			{PageContent: "Enterprise plans include SSO and priority support."},
		},
	}

	query := "What is the refund policy?"

	dense := engine.NewRAGEngine(store, nil, nil, engine.Config{
		TopK:          2,
		MinScore:      0.7,
		ForceAlways:   true,
		RetrievalMode: "dense",
	})
	denseBlock, denseUsed, denseErr := dense.Query(ctx, query)

	reranked := engine.NewRAGEngine(store, keywordReranker{}, nil, engine.Config{
		TopK:          2,
		RetrievalMode: "reranker",
		RerankerK:     4,
		ForceAlways:   true,
	})
	rerankBlock, rerankUsed, rerankErr := reranked.Query(ctx, query)

	fmt.Println("== Dense ==")
	fmt.Printf("used=%v err=%v\n%s\n\n", denseUsed, denseErr, denseBlock)

	fmt.Println("== Reranker ==")
	fmt.Printf("used=%v err=%v\n%s\n", rerankUsed, rerankErr, rerankBlock)
}
