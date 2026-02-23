package store

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tmc/langchaingo/embeddings"
	vectorstores "github.com/tmc/langchaingo/vectorstores"
	qdrantstore "github.com/tmc/langchaingo/vectorstores/qdrant"

	"gorag/engine/models"
)

const (
	qdrantDistanceCosine = "Cosine"
	qdrantRequestTimeout = 15 * time.Second
)

var (
	ErrNilEmbedder          = errors.New("embedder cannot be nil")
	ErrEmptyCollection      = errors.New("qdrant collection name cannot be empty")
	ErrEmptyVectorSize      = errors.New("vector size must be greater than zero")
	ErrEmptyQdrantURL       = errors.New("qdrant URL cannot be empty")
	ErrEmptyEmbeddingVector = errors.New("embedder returned an empty vector")
)

// BootstrapResult holds the outcome of preparing a Qdrant collection.
type BootstrapResult struct {
	PointsCount int
	ShouldIndex bool
}

// NewQdrant creates a ready-to-use Qdrant vector store.
func NewQdrant(
	rawURL, apiKey, collection string,
	embedder embeddings.Embedder,
) (vectorstores.VectorStore, error) {
	if embedder == nil {
		return nil, ErrNilEmbedder
	}

	baseURL, err := parseQdrantURL(rawURL)
	if err != nil {
		return nil, err
	}

	opts := []qdrantstore.Option{
		qdrantstore.WithURL(*baseURL),
		qdrantstore.WithCollectionName(strings.TrimSpace(collection)),
		qdrantstore.WithEmbedder(embedder),
	}
	if key := strings.TrimSpace(apiKey); key != "" {
		opts = append(opts, qdrantstore.WithAPIKey(key))
	}

	store, err := qdrantstore.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("error initializing qdrant vector store: %w", err)
	}

	return store, nil
}

// PrepareCollection ensures the Qdrant collection exists and is ready for indexing.
// If reIndexOnBoot is true, any existing collection is deleted and recreated.
func PrepareCollection(
	ctx context.Context,
	rawURL, apiKey, collection string,
	vectorSize int,
	reIndexOnBoot bool,
) (BootstrapResult, error) {
	if vectorSize <= 0 {
		return BootstrapResult{}, ErrEmptyVectorSize
	}

	collection = strings.TrimSpace(collection)
	if collection == "" {
		return BootstrapResult{}, ErrEmptyCollection
	}

	baseURL, err := parseQdrantURL(rawURL)
	if err != nil {
		return BootstrapResult{}, err
	}

	client := &http.Client{Timeout: qdrantRequestTimeout}

	if reIndexOnBoot {
		if err := deleteCollection(ctx, client, baseURL, collection, apiKey); err != nil {
			return BootstrapResult{}, err
		}
	}

	exists, pointsCount, err := collectionState(ctx, client, baseURL, collection, apiKey)
	if err != nil {
		return BootstrapResult{}, err
	}

	if !exists {
		if err := createCollection(ctx, client, baseURL, collection, apiKey, vectorSize); err != nil {
			return BootstrapResult{}, err
		}
		return BootstrapResult{ShouldIndex: true}, nil
	}

	if pointsCount > 0 && !reIndexOnBoot {
		return BootstrapResult{PointsCount: pointsCount, ShouldIndex: false}, nil
	}

	return BootstrapResult{ShouldIndex: true}, nil
}

// ProbeEmbeddingSize returns the vector dimension produced by the embedder,
// using the first non-empty chunk content as the sample text.
func ProbeEmbeddingSize(ctx context.Context, embedder embeddings.Embedder, chunks []models.KnowledgeChunk) (int, error) {
	if embedder == nil {
		return 0, ErrNilEmbedder
	}

	sample := "dimension probe"
	for _, chunk := range chunks {
		if content := strings.TrimSpace(chunk.Content); content != "" {
			sample = content
			break
		}
	}

	vec, err := embedder.EmbedQuery(ctx, sample)
	if err != nil {
		return 0, fmt.Errorf("error probing embedding dimension: %w", err)
	}
	if len(vec) == 0 {
		return 0, ErrEmptyEmbeddingVector
	}

	return len(vec), nil
}

// collectionInfo maps the relevant fields from the Qdrant collection info response.
type collectionInfo struct {
	Result struct {
		PointsCount  int `json:"points_count"`
		VectorsCount int `json:"vectors_count"`
	} `json:"result"`
}

// collectionState checks whether a collection exists and returns its point count.
func collectionState(ctx context.Context, client *http.Client, baseURL *url.URL, collection, apiKey string) (bool, int, error) {
	endpoint := baseURL.JoinPath("collections", collection)
	body, status, err := doRequest(ctx, client, http.MethodGet, endpoint, apiKey, nil)
	if err != nil {
		return false, 0, err
	}

	switch status {
	case http.StatusNotFound:
		return false, 0, nil
	case http.StatusOK:
		var info collectionInfo
		if err := json.Unmarshal(body, &info); err != nil {
			return false, 0, fmt.Errorf("error parsing qdrant collection state: %w", err)
		}
		points := info.Result.PointsCount
		if points == 0 {
			points = info.Result.VectorsCount
		}
		return true, points, nil
	default:
		return false, 0, fmt.Errorf("unexpected qdrant status %d: %s", status, body)
	}
}

// createCollection creates a new Qdrant collection with the given vector size.
func createCollection(ctx context.Context, client *http.Client, baseURL *url.URL, collection, apiKey string, vectorSize int) error {
	endpoint := baseURL.JoinPath("collections", collection)
	payload := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": qdrantDistanceCosine,
		},
	}

	body, status, err := doRequest(ctx, client, http.MethodPut, endpoint, apiKey, payload)
	if err != nil {
		return err
	}
	if status != http.StatusOK && status != http.StatusCreated {
		return fmt.Errorf("error creating qdrant collection (status %d): %s", status, body)
	}
	return nil
}

// deleteCollection removes a Qdrant collection, ignoring 404 responses.
func deleteCollection(ctx context.Context, client *http.Client, baseURL *url.URL, collection, apiKey string) error {
	endpoint := baseURL.JoinPath("collections", collection)
	body, status, err := doRequest(ctx, client, http.MethodDelete, endpoint, apiKey, nil)
	if err != nil {
		return err
	}
	if status == http.StatusOK || status == http.StatusNotFound {
		return nil
	}
	return fmt.Errorf("error deleting qdrant collection (status %d): %s", status, body)
}

// doRequest executes an HTTP request against the Qdrant API and returns the raw response body and status code.
func doRequest(ctx context.Context, client *http.Client, method string, endpoint *url.URL, apiKey string, payload any) ([]byte, int, error) {
	var bodyReader io.Reader
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return nil, 0, fmt.Errorf("error marshaling qdrant payload: %w", err)
		}
		bodyReader = bytes.NewReader(raw)
	}

	req, err := http.NewRequestWithContext(ctx, method, endpoint.String(), bodyReader)
	if err != nil {
		return nil, 0, fmt.Errorf("error creating qdrant request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key := strings.TrimSpace(apiKey); key != "" {
		req.Header.Set("api-key", key)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("qdrant communication error: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("error reading qdrant response: %w", err)
	}

	return respBody, resp.StatusCode, nil
}

// parseQdrantURL validates and parses a raw Qdrant URL string,
// prepending "http://" if no scheme is present.
func parseQdrantURL(rawURL string) (*url.URL, error) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil, ErrEmptyQdrantURL
	}
	if !strings.Contains(rawURL, "://") {
		rawURL = "http://" + rawURL
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("invalid qdrant URL: %w", err)
	}
	if strings.TrimSpace(u.Scheme) == "" || strings.TrimSpace(u.Host) == "" {
		return nil, fmt.Errorf("invalid qdrant URL: scheme and host are required")
	}

	return u, nil
}
