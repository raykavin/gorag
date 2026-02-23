package store

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/raykavin/gorag/pkg/models"
)

type testEmbedder struct {
	query func(ctx context.Context, text string) ([]float32, error)
}

func (t testEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if t.query != nil {
		return t.query(ctx, text)
	}
	return []float32{1, 2}, nil
}

func (t testEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 2}
	}
	return out, nil
}

type qdrantRT struct {
	mu          sync.Mutex
	state       string
	getCalls    int
	putCalls    int
	deleteCalls int
}

func (rt *qdrantRT) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	status := http.StatusOK
	body := "{}"

	switch req.Method {
	case http.MethodDelete:
		rt.deleteCalls++
		rt.state = "missing"
	case http.MethodGet:
		rt.getCalls++
		if rt.state == "missing" {
			status = http.StatusNotFound
			body = ""
		} else {
			body = `{"result":{"points_count":4}}`
		}
	case http.MethodPut:
		rt.putCalls++
		rt.state = "created"
		status = http.StatusCreated
	default:
		status = http.StatusMethodNotAllowed
	}

	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}, nil
}

func TestParseQdrantURL(t *testing.T) {
	if _, err := parseQdrantURL(" "); !errors.Is(err, ErrEmptyQdrantURL) {
		t.Fatalf("expected ErrEmptyQdrantURL, got %v", err)
	}
	if _, err := parseQdrantURL("http://"); err == nil {
		t.Fatal("expected invalid URL error")
	}

	u, err := parseQdrantURL("localhost:6333")
	if err != nil {
		t.Fatal(err)
	}
	if u.Scheme != "http" || u.Host != "localhost:6333" {
		t.Fatalf("unexpected parsed URL: %s", u.String())
	}
}

func TestProbeEmbeddingSize(t *testing.T) {
	if _, err := ProbeEmbeddingSize(context.Background(), nil, nil); !errors.Is(err, ErrNilEmbedder) {
		t.Fatalf("expected ErrNilEmbedder, got %v", err)
	}

	emb := testEmbedder{query: func(ctx context.Context, text string) ([]float32, error) {
		if text != "chunk content" {
			t.Fatalf("unexpected probe text: %q", text)
		}
		return []float32{1, 2, 3}, nil
	}}
	got, err := ProbeEmbeddingSize(context.Background(), emb, []models.KnowledgeChunk{{Content: " "}, {Content: "chunk content"}})
	if err != nil {
		t.Fatal(err)
	}
	if got != 3 {
		t.Fatalf("expected size 3, got %d", got)
	}

	empty := testEmbedder{query: func(ctx context.Context, text string) ([]float32, error) { return nil, nil }}
	if _, err := ProbeEmbeddingSize(context.Background(), empty, nil); !errors.Is(err, ErrEmptyEmbeddingVector) {
		t.Fatalf("expected ErrEmptyEmbeddingVector, got %v", err)
	}
}

func TestPrepareCollectionFlows(t *testing.T) {
	if _, err := PrepareCollection(context.Background(), "http://localhost:6333", "", "col", 0, false); !errors.Is(err, ErrEmptyVectorSize) {
		t.Fatalf("expected ErrEmptyVectorSize, got %v", err)
	}
	if _, err := PrepareCollection(context.Background(), "http://localhost:6333", "", " ", 3, false); !errors.Is(err, ErrEmptyCollection) {
		t.Fatalf("expected ErrEmptyCollection, got %v", err)
	}

	orig := http.DefaultTransport
	rt := &qdrantRT{state: "missing"}
	http.DefaultTransport = rt
	t.Cleanup(func() { http.DefaultTransport = orig })

	res, err := PrepareCollection(context.Background(), "http://qdrant.local", "", "col", 3, false)
	if err != nil {
		t.Fatal(err)
	}
	if !res.ShouldIndex || rt.putCalls != 1 || rt.getCalls == 0 {
		t.Fatalf("unexpected bootstrap result: %+v, calls get=%d put=%d", res, rt.getCalls, rt.putCalls)
	}

	res, err = PrepareCollection(context.Background(), "http://qdrant.local", "", "col", 3, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.ShouldIndex || res.PointsCount != 4 {
		t.Fatalf("expected reuse existing collection, got %+v", res)
	}

	res, err = PrepareCollection(context.Background(), "http://qdrant.local", "", "col", 3, true)
	if err != nil {
		t.Fatal(err)
	}
	if !res.ShouldIndex || rt.deleteCalls == 0 {
		t.Fatalf("expected reindex flow, got %+v deleteCalls=%d", res, rt.deleteCalls)
	}
}

func TestQdrantRequestHelpers(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.Header.Get("api-key") != "k" {
			t.Fatalf("missing api-key header")
		}
		if req.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", req.Method)
		}
		raw, _ := io.ReadAll(req.Body)
		if !strings.Contains(string(raw), `"x":1`) {
			t.Fatalf("unexpected payload: %s", string(raw))
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"ok":true}`)), Request: req}, nil
	})}

	u, _ := url.Parse("http://qdrant.local")
	body, status, err := doRequest(context.Background(), client, http.MethodPost, u, "k", map[string]any{"x": 1})
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusOK || !strings.Contains(string(body), "ok") {
		t.Fatalf("unexpected response: status=%d body=%s", status, string(body))
	}

	if _, err := NewQdrant("http://localhost:6333", "", "col", nil); !errors.Is(err, ErrNilEmbedder) {
		t.Fatalf("expected ErrNilEmbedder, got %v", err)
	}
	if _, err := NewQdrant("::bad", "", "col", testEmbedder{}); err == nil {
		t.Fatal("expected invalid URL error")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestCollectionStateAndCreateDeleteErrors(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		status := http.StatusInternalServerError
		body := `{"error":"bad"}`
		if req.Method == http.MethodGet {
			status = http.StatusOK
			body = `{"result":{"vectors_count":2}}`
		}
		return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(bytes.NewBufferString(body)), Request: req}, nil
	})}
	u, _ := url.Parse("http://qdrant.local")

	exists, points, err := collectionState(context.Background(), client, u, "c", "")
	if err != nil || !exists || points != 2 {
		t.Fatalf("unexpected collection state: exists=%v points=%d err=%v", exists, points, err)
	}

	if err := createCollection(context.Background(), client, u, "c", "", 3); err == nil {
		t.Fatal("expected createCollection status error")
	}
	if err := deleteCollection(context.Background(), client, u, "c", ""); err == nil {
		t.Fatal("expected deleteCollection status error")
	}

	if _, _, err := doRequest(context.Background(), client, http.MethodPost, u, "", func() {}); err == nil {
		t.Fatal("expected marshal error")
	}

	badClient := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(errReader{}), Request: req}, nil
	})}
	if _, _, err := doRequest(context.Background(), badClient, http.MethodGet, u, "", nil); err == nil {
		t.Fatal("expected read body error")
	}

	if _, err := parseQdrantURL("http://"); err == nil {
		t.Fatal("expected invalid URL error")
	}

	data, _ := json.Marshal(BootstrapResult{PointsCount: 1, ShouldIndex: true})
	if !strings.Contains(string(data), "PointsCount") {
		t.Fatal("expected default json encoding for bootstrap result")
	}
}

type errReader struct{}

func (errReader) Read(p []byte) (int, error) { return 0, errors.New("read error") }
func (errReader) Close() error               { return nil }
