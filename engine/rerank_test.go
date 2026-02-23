package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/schema"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestNewCrossEncoderRerankerValidation(t *testing.T) {
	if _, err := NewCrossEncoderReranker(CrossEncoderConfig{Model: "m"}); !errors.Is(err, ErrEmptyRerankerBaseURL) {
		t.Fatalf("expected ErrEmptyRerankerBaseURL, got %v", err)
	}
	if _, err := NewCrossEncoderReranker(CrossEncoderConfig{BaseURL: "http://x"}); !errors.Is(err, ErrEmptyRerankerModel) {
		t.Fatalf("expected ErrEmptyRerankerModel, got %v", err)
	}
}

func TestResolveRerankerEndpoint(t *testing.T) {
	got, err := resolveRerankerEndpoint("http://localhost:8000", "")
	if err != nil {
		t.Fatal(err)
	}
	if got != "http://localhost:8000/api/rerank" {
		t.Fatalf("unexpected endpoint: %s", got)
	}

	got, err = resolveRerankerEndpoint("http://localhost:8000", "https://api.example.com/custom")
	if err != nil {
		t.Fatal(err)
	}
	if got != "https://api.example.com/custom" {
		t.Fatalf("unexpected absolute endpoint: %s", got)
	}
}

func TestCrossEncoderRerankerRerank(t *testing.T) {
	var seenAuth string
	var seenTopN int

	r, err := NewCrossEncoderReranker(CrossEncoderConfig{
		BaseURL: "http://reranker.local",
		Model:   "bge-reranker",
		APIKey:  "secret",
	})
	if err != nil {
		t.Fatal(err)
	}
	r.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		seenAuth = req.Header.Get("Authorization")
		body, _ := io.ReadAll(req.Body)
		_ = req.Body.Close()

		var payload rerankRequest
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, err
		}
		seenTopN = payload.TopN

		respBody, _ := json.Marshal(rerankResponse{Results: []rerankResult{{Index: 1, RelevanceScore: 0.8}, {Index: 0, Score: 0.5}}})
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(bytes.NewReader(respBody)),
			Header:     make(http.Header),
		}, nil
	})

	docs := []schema.Document{{PageContent: "doc0"}, {PageContent: "doc1"}}
	got, err := r.Rerank(context.Background(), "query", docs, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].PageContent != "doc1" {
		t.Fatalf("unexpected rerank output: %+v", got)
	}
	if seenAuth != "Bearer secret" {
		t.Fatalf("expected auth header, got %q", seenAuth)
	}
	if seenTopN != 1 {
		t.Fatalf("expected top_n=1, got %d", seenTopN)
	}
}

func TestCrossEncoderRerankerErrorPaths(t *testing.T) {
	if _, err := (&CrossEncoderReranker{}).Rerank(context.Background(), "q", nil, 1); !errors.Is(err, ErrUninitializedReranker) {
		t.Fatalf("expected ErrUninitializedReranker, got %v", err)
	}

	r, err := NewCrossEncoderReranker(CrossEncoderConfig{BaseURL: "http://reranker.local", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	r.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"results":[]}`)),
			Header:     make(http.Header),
		}, nil
	})

	_, err = r.Rerank(context.Background(), "q", []schema.Document{{PageContent: "doc"}}, 1)
	if !errors.Is(err, ErrNoRerankerResults) {
		t.Fatalf("expected ErrNoRerankerResults, got %v", err)
	}

	if _, err := pickTopDocs([]rerankResult{{Index: 5, Score: 1}}, []schema.Document{{PageContent: "a"}}, 1); !errors.Is(err, ErrInvalidRerankerIndex) {
		t.Fatalf("expected ErrInvalidRerankerIndex, got %v", err)
	}

	filteredDocs, filteredTexts := filterDocuments([]schema.Document{{PageContent: " "}, {PageContent: "x"}})
	if len(filteredDocs) != 1 || len(filteredTexts) != 1 || filteredTexts[0] != "x" {
		t.Fatalf("unexpected filtered output: %+v %+v", filteredDocs, filteredTexts)
	}

	if score := rerankScore(rerankResult{Score: 0.1, RelevanceScore: 0.7}); score != 0.7 {
		t.Fatalf("unexpected rerank score: %v", score)
	}

	if _, err := resolveRerankerEndpoint("::bad", ""); err == nil || !strings.Contains(err.Error(), "invalid reranker base URL") {
		t.Fatalf("expected invalid base url error, got %v", err)
	}
}
