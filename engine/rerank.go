package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/tmc/langchaingo/schema"
)

const (
	defRerankerEndpointPath = "/api/rerank"
	defRerankerTimeout      = 20 * time.Second
)

var (
	ErrEmptyRerankerBaseURL  = errors.New("reranker base URL cannot be empty")
	ErrEmptyRerankerModel    = errors.New("reranker model cannot be empty")
	ErrUninitializedReranker = errors.New("reranker not initialized")
	ErrNoRerankerResults     = errors.New("reranker returned no results")
	ErrInvalidRerankerIndex  = errors.New("reranker returned only invalid indexes")
)

// CrossEncoderConfig holds the configuration for a CrossEncoderReranker.
type CrossEncoderConfig struct {
	BaseURL      string
	EndpointPath string
	APIKey       string
	Model        string
	Timeout      time.Duration
}

// CrossEncoderReranker reranks documents using a cross-encoder HTTP API.
type CrossEncoderReranker struct {
	client   *http.Client
	endpoint string
	apiKey   string
	model    string
}

// NewCrossEncoderReranker creates a CrossEncoderReranker from the given config.
func NewCrossEncoderReranker(cfg CrossEncoderConfig) (*CrossEncoderReranker, error) {
	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		return nil, ErrEmptyRerankerBaseURL
	}

	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		return nil, ErrEmptyRerankerModel
	}

	endpoint, err := resolveRerankerEndpoint(baseURL, cfg.EndpointPath)
	if err != nil {
		return nil, err
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = defRerankerTimeout
	}

	return &CrossEncoderReranker{
		client:   &http.Client{Timeout: timeout},
		endpoint: endpoint,
		apiKey:   strings.TrimSpace(cfg.APIKey),
		model:    model,
	}, nil
}

// Rerank sends documents to the cross-encoder API and returns the top-K results
// sorted by relevance score in descending order.
func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, docs []schema.Document, topK int) ([]schema.Document, error) {
	if r == nil || r.client == nil {
		return nil, ErrUninitializedReranker
	}

	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}

	filtered, texts := filterDocuments(docs)
	if len(filtered) == 0 {
		return nil, nil
	}

	if topK <= 0 || topK > len(filtered) {
		topK = len(filtered)
	}

	results, err := r.callAPI(ctx, query, texts, topK)
	if err != nil {
		return nil, err
	}

	return pickTopDocs(results, filtered, topK)
}

// callAPI serializes the request, calls the rerank endpoint and returns parsed results.
func (r *CrossEncoderReranker) callAPI(ctx context.Context, query string, texts []string, topK int) ([]rerankResult, error) {
	payload, err := json.Marshal(rerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: texts,
		TopN:      topK,
	})
	if err != nil {
		return nil, fmt.Errorf("error marshaling rerank request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("error creating rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if r.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
		req.Header.Set("api-key", r.apiKey)
	}

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("reranker communication error: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading reranker response: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("reranker returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var parsed rerankResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("error parsing reranker response: %w", err)
	}

	results := parsed.Results
	if len(results) == 0 {
		results = parsed.Data
	}
	if len(results) == 0 {
		return nil, ErrNoRerankerResults
	}

	return results, nil
}

// pickTopDocs sorts results by score and maps valid indexes back to documents.
func pickTopDocs(results []rerankResult, docs []schema.Document, topK int) ([]schema.Document, error) {
	sort.SliceStable(results, func(i, j int) bool {
		si, sj := rerankScore(results[i]), rerankScore(results[j])
		if si == sj {
			return results[i].Index < results[j].Index
		}
		return si > sj
	})

	out := make([]schema.Document, 0, topK)
	seen := make(map[int]struct{}, len(results))

	for _, res := range results {
		if len(out) >= topK {
			break
		}
		if res.Index < 0 || res.Index >= len(docs) {
			continue
		}
		if _, ok := seen[res.Index]; ok {
			continue
		}
		seen[res.Index] = struct{}{}
		out = append(out, docs[res.Index])
	}

	if len(out) == 0 {
		return nil, ErrInvalidRerankerIndex
	}
	return out, nil
}

// filterDocuments returns docs and their text content, skipping empty entries.
func filterDocuments(docs []schema.Document) ([]schema.Document, []string) {
	filtered := make([]schema.Document, 0, len(docs))
	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		if content := strings.TrimSpace(doc.PageContent); content != "" {
			filtered = append(filtered, doc)
			texts = append(texts, content)
		}
	}
	return filtered, texts
}

// resolveRerankerEndpoint builds the final rerank endpoint URL from a base URL and optional path.
func resolveRerankerEndpoint(baseURL, endpointPath string) (string, error) {
	path := strings.TrimSpace(endpointPath)
	if path == "" {
		path = defRerankerEndpointPath
	}

	// If the path is already a full URL, validate and return it directly.
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		parsed, err := url.Parse(path)
		if err != nil {
			return "", fmt.Errorf("invalid reranker endpoint: %w", err)
		}
		if parsed.Scheme == "" || parsed.Host == "" {
			return "", fmt.Errorf("invalid reranker endpoint: %s", path)
		}
		return parsed.String(), nil
	}

	base, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid reranker base URL: %w", err)
	}
	if base.Scheme == "" || base.Host == "" {
		return "", fmt.Errorf("invalid reranker base URL: %s", baseURL)
	}

	return base.JoinPath(strings.TrimPrefix(path, "/")).String(), nil
}
