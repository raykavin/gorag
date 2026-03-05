# gorag

[![Go Reference](https://pkg.go.dev/badge/github.com/raykavin/gorag.svg)](https://pkg.go.dev/github.com/raykavin/gorag)
[![Go Version](https://img.shields.io/badge/go-1.21+-blue)](https://golang.org/dl/)
[![Go Report Card](https://goreportcard.com/badge/github.com/raykavin/gorag)](https://goreportcard.com/report/github.com/raykavin/gorag)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

A Go library for building Retrieval-Augmented Generation (RAG) flows with:
- knowledge loading and chunking
- embedding and response caching
- Qdrant vector store bootstrap helpers
- dense retrieval with optional reranking

Module path: `github.com/raykavin/gorag`

## Packages

- `engine`: RAG pipeline building blocks
- `cache`: in-memory embedding and response caches
- `pkg/store`: Qdrant bootstrap and vector store setup helpers
- `pkg/models`: shared data models (`KnowledgeChunk`, `ChatResponse`, `ToolCallLog`)

## Install

```bash
go get github.com/raykavin/gorag
```

## Quick Start

```go
package main

import (
	"context"
	"log"
	"time"

	"github.com/raykavin/gorag/cache"
	"github.com/raykavin/gorag/engine"
	"github.com/raykavin/gorag/pkg/store"
	"github.com/tmc/langchaingo/embeddings"
)

func main() {
	ctx := context.Background()

	// Create your base embedder (provider-specific).
	var base embeddings.Embedder

	// Optional embedding cache wrapper.
	embCache := cache.NewEmbedding(30*time.Minute, 10*time.Minute)
	cachedEmbedder, err := engine.NewCacheEmbedder(base, embCache)
	if err != nil {
		log.Fatal(err)
	}

	// Probe vector size from your chunks.
	vectorSize, err := store.ProbeEmbeddingSize(ctx, cachedEmbedder, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Ensure Qdrant collection exists.
	_, err = store.PrepareCollection(
		ctx,
		"http://localhost:6333",
		"",          // api key
		"knowledge", // collection
		vectorSize,
		false,       // reIndexOnBoot
	)
	if err != nil {
		log.Fatal(err)
	}

	// Build vector store.
	vs, err := store.NewQdrant("http://localhost:6333", "", "knowledge", cachedEmbedder)
	if err != nil {
		log.Fatal(err)
	}

	// Build RAG engine.
	rag := engine.NewRAGEngine(vs, nil, embCache, engine.Config{
		TopK:          3,
		MinScore:      0.7,
		ForceAlways:   true,
		RetrievalMode: "dense", // or "reranker"
		RerankerK:     8,
	})

	contextBlock, used, err := rag.Query(ctx, "What is our refund policy?")
	if err != nil {
		log.Fatal(err)
	}
	if used {
		log.Println(contextBlock)
	}
}
```

## Knowledge Loading

Use `engine.ReadKnowledge(path)` to read and expand JSON knowledge files.

Supported sources:
- inline chunk `content`
- explicit file list in `files`
- directory traversal via `files_path`

Supported file extensions:
- `.txt`, `.md`, `.markdown`, `.csv`

Example input shape (`pkg/models.KnowledgeChunk`):

```json
[
  {
    "id": "KB_001",
    "category": "refunds",
    "files": ["docs/refunds.md"], // Load files from list
  },
  {
	"id": "KB_002",
    "category": "terms_and_conditions",
    "content": "The terms and conditions are...", // Inline chunk
  },
  {
    "id": "KB_003",
	"category": "faq",
    "files_path": "docs/faq" // Load all files in directory
  }
]
```

## Retrieval Modes

`engine.RAGEngine` supports:

- `dense`:
  - similarity search with score threshold (`MinScore`)
  - optional fallback search without threshold if `ForceAlways=true`

- `reranker`:
  - dense candidate retrieval (`RerankerK`)
  - optional rerank step to return top `TopK`

## Reranker

Use `engine.NewCrossEncoderReranker(engine.CrossEncoderConfig{...})`.

Important options:
- `BaseURL`: reranker API base URL (required)
- `EndpointPath`: defaults to `/api/rerank`
- `Model`: reranker model name (required)
- `APIKey`: optional (sent as `Authorization: Bearer ...` and `api-key`)
- `Timeout`: defaults to `20s`

## Cache

### Embedding Cache

`cache.Embedding`:
- key: SHA-256(text)
- methods: `Get`, `Set`, `Stats`

### Response Cache

`cache.ResponseCache`:
- key: normalized `(text + historyKey)` hash
- methods: `Get`, `Set`, `Invalidate`, `Stats`
- bounded size with oldest-item eviction
- TTL policy based on tool names used in `ChatResponse.ToolCalls`

## Qdrant Helpers

`pkg/store` provides:
- `PrepareCollection`: create/check/delete collection on boot
- `NewQdrant`: build langchaingo Qdrant vector store
- `ProbeEmbeddingSize`: infer vector dimension from embedder output

## Error Handling

Packages expose typed errors for common validation/setup failures (for example, nil embedders, empty URLs, invalid reranker config). Prefer `errors.Is` checks when branching.

## Testing

Run all tests:

```bash
go test ./...
```

If your environment restricts default Go cache paths, run:

```bash
GOCACHE=/tmp/go-build go test ./...
```

---
## Contributing

Contributions to gorag are welcome! Here are some ways you can help improve the project:

- **Report bugs and suggest features** by opening issues on GitHub
- **Submit pull requests** with bug fixes or new features
- **Improve documentation** to help other users and developers
- **Share your custom strategies** with the community

## License

gorag is distributed under the **MIT License**.  
For complete license terms and conditions, see the [LICENSE](LICENSE.md) file in the repository.

---

## Contact

For support, collaboration, or questions about gorag:

**Email**: [raykavin.meireles@gmail.com](mailto:raykavin.meireles@gmail.com)  
**GitHub**: [@raykavin](https://github.com/raykavin)  
