package engine

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/raykavin/gorag/pkg/models"
)

func TestReadKnowledgeErrors(t *testing.T) {
	_, err := ReadKnowledge("/definitely/missing.json")
	if !errors.Is(err, ErrReadingKnowledgeFile) {
		t.Fatalf("expected ErrReadingKnowledgeFile, got %v", err)
	}

	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = ReadKnowledge(bad)
	if !errors.Is(err, ErrParsingKnowledgeFile) {
		t.Fatalf("expected ErrParsingKnowledgeFile, got %v", err)
	}
}

func TestReadKnowledgeExpandsInlineAndFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("line1\nline2"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "docs", "a.md"), []byte("doc-a"), 0o600); err != nil {
		t.Fatal(err)
	}

	data := []models.KnowledgeChunk{
		{
			ID:        "K1",
			Category:  "cat",
			Content:   "inline",
			Role:      "system",
			Files:     []string{"note.txt"},
			FilesPath: "docs",
		},
		{
			ID:       "K2",
			Category: "cat",
			Content:  "no-role-default",
		},
	}

	raw, _ := json.Marshal(data)
	path := filepath.Join(dir, "knowledge.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}

	chunks, err := ReadKnowledge(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected expanded chunks, got %d", len(chunks))
	}
	if CountIndexableChunks(chunks) != len(chunks) {
		t.Fatalf("expected all chunks indexable: %+v", chunks)
	}

	hasInline := false
	hasDefaultRole := false
	for _, c := range chunks {
		if c.Content == "inline" {
			hasInline = true
		}
		if strings.HasPrefix(c.ID, "K1") && c.Role != "system" {
			t.Fatalf("expected role system for K1 chunks, got %q (chunk=%+v)", c.Role, c)
		}
		if c.Content == "no-role-default" {
			hasDefaultRole = true
			if c.Role != "user" {
				t.Fatalf("expected default role user, got %q", c.Role)
			}
		}
	}
	if !hasInline {
		t.Fatal("expected inline content chunk")
	}
	if !hasDefaultRole {
		t.Fatal("expected no-role-default chunk")
	}
}

func TestLoaderHelpers(t *testing.T) {
	if !hasLikelyHeader([]string{"name", "age"}) {
		t.Fatal("expected header")
	}
	if hasLikelyHeader([]string{"name", "age2"}) {
		t.Fatal("did not expect header")
	}

	line := stringifyCSVRow(3, []string{"A", "", "B"}, []string{"c1", "c2", ""}, true)
	if !strings.Contains(line, "row 3") || !strings.Contains(line, "c1: A") || !strings.Contains(line, "column 3: B") {
		t.Fatalf("unexpected csv line: %q", line)
	}

	chunks := splitTextIntoChunks(strings.Repeat("a", 15), 10, 2)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	if got := normalizeLineBreaks("a\r\nb\rc"); got != "a\nb\nc" {
		t.Fatalf("unexpected line break normalization: %q", got)
	}

	if id := buildChunkID("KB", "file", 0, 0, 2); id != "KB_f001_file_001" {
		t.Fatalf("unexpected chunk id: %s", id)
	}
	if p := sanitizeIDPart("Docs/My File.md"); p != "docs_my_file" {
		t.Fatalf("unexpected sanitized id: %s", p)
	}
	if !isSupportedExtension("a.markdown") || isSupportedExtension("a.pdf") {
		t.Fatal("extension support mismatch")
	}
}

func TestCollectSupportedFilesAndLoadCSV(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(csvPath, []byte("name,age\nalice,10\nbob,20\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "skip.bin"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	files, err := collectSupportedFiles(dir, ".")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 || filepath.Base(files[0]) != "data.csv" {
		t.Fatalf("unexpected supported files: %v", files)
	}

	rows, err := loadCSVChunks(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || !strings.Contains(rows[0], "name: alice") {
		t.Fatalf("unexpected csv chunks: %v", rows)
	}
}
