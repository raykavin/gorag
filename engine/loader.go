package engine

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/raykavin/gorag/pkg/models"
)

var (
	ErrReadingKnowledgeFile = errors.New("error reading knowledge file")
	ErrParsingKnowledgeFile = errors.New("error parsing knowledge file")
)

const (
	defKnowledgeRole = "user"
	roleSystem       = "system"
)

// ReadKnowledge reads and parses a knowledge base JSON file, returning expanded chunks.
func ReadKnowledge(path string) ([]models.KnowledgeChunk, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrReadingKnowledgeFile, err)
	}

	var chunks []models.KnowledgeChunk
	if err := json.Unmarshal(content, &chunks); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrParsingKnowledgeFile, err)
	}

	return expandKnowledgeChunks(chunks, filepath.Dir(path))
}

// CountIndexableChunks returns the number of non-empty chunks.
func CountIndexableChunks(chunks []models.KnowledgeChunk) int {
	count := 0
	for _, c := range chunks {
		if strings.TrimSpace(c.Content) != "" {
			count++
		}
	}
	return count
}

// expandKnowledgeChunks expands inline content and file references into individual chunks.
func expandKnowledgeChunks(chunks []models.KnowledgeChunk, baseDir string) ([]models.KnowledgeChunk, error) {
	expanded := make([]models.KnowledgeChunk, 0, len(chunks))

	for i, chunk := range chunks {
		baseID := strings.TrimSpace(chunk.ID)
		if baseID == "" {
			baseID = fmt.Sprintf("KB_%03d", i+1)
		}

		category := strings.TrimSpace(chunk.Category)
		role := normalizeKnowledgeRole(chunk.Role)

		if content := strings.TrimSpace(chunk.Content); content != "" {
			expanded = append(expanded, models.KnowledgeChunk{
				ID:       baseID,
				Category: category,
				Role:     role,
				Content:  content,
				Source:   strings.TrimSpace(chunk.Source),
			})
		}

		filePaths, err := resolveChunkFiles(chunk, baseDir)
		if err != nil {
			return nil, fmt.Errorf("item %q: %w", baseID, err)
		}

		fileChunks, err := expandFilePaths(filePaths, baseID, category, role, baseDir)
		if err != nil {
			return nil, fmt.Errorf("item %q: %w", baseID, err)
		}

		expanded = append(expanded, fileChunks...)
	}

	return expanded, nil
}

// resolveChunkFiles collects the ordered list of absolute file paths for a chunk,
// combining explicitly listed files with those discovered under FilesPath.
func resolveChunkFiles(chunk models.KnowledgeChunk, baseDir string) ([]string, error) {
	var paths []string

	for _, fRef := range chunk.Files {
		fRef = strings.TrimSpace(fRef)
		if fRef != "" {
			paths = append(paths, resolveKnowledgePath(baseDir, fRef))
		}
	}

	if filesPath := strings.TrimSpace(chunk.FilesPath); filesPath != "" {
		collected, err := collectSupportedFiles(baseDir, filesPath)
		if err != nil {
			return nil, fmt.Errorf("files_path %q: %w", filesPath, err)
		}
		paths = append(paths, collected...)
	}

	return paths, nil
}

// expandFilePaths loads chunks from each file path and converts them to
// KnowledgeChunk records with stable, indexed IDs.
func expandFilePaths(
	paths []string,
	baseID, category, role, baseDir string,
) ([]models.KnowledgeChunk, error) {
	var expanded []models.KnowledgeChunk

	for fileIdx, fPath := range paths {
		source := relativePathOrOriginal(baseDir, fPath)
		fileIDPart := sanitizeIDPart(source)

		textChunks, err := loadFileChunks(fPath)
		if err != nil {
			return nil, fmt.Errorf("file %q: %w", source, err)
		}

		for chunkIdx, text := range textChunks {
			text = strings.TrimSpace(text)
			if text == "" {
				continue
			}

			chunkID := buildChunkID(
				baseID,
				fileIDPart,
				fileIdx,
				chunkIdx,
				len(textChunks),
			)

			expanded = append(expanded, models.KnowledgeChunk{
				ID:       chunkID,
				Category: category,
				Role:     role,
				Content:  text,
				Source:   source,
			})
		}
	}

	return expanded, nil
}

// collectSupportedFiles returns all supported files from a path (file or directory).
func collectSupportedFiles(baseDir, pathRef string) ([]string, error) {
	root := resolveKnowledgePath(baseDir, pathRef)

	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		if !isSupportedExtension(root) {
			return nil, fmt.Errorf("unsupported file extension: %q", pathRef)
		}
		return []string{root}, nil
	}

	var files []string
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() && isSupportedExtension(path) {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	sort.Strings(files)

	if len(files) == 0 {
		return nil, fmt.Errorf("no supported files found in %q", pathRef)
	}

	return files, nil
}

// loadFileChunks dispatches to the appropriate loader based on file extension.
func loadFileChunks(path string) ([]string, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".csv":
		return loadCSVChunks(path)
	case ".txt", ".md", ".markdown":
		return loadTextChunks(path)
	default:
		return nil, fmt.Errorf("unsupported extension for indexing: %s", filepath.Ext(path))
	}
}

// loadTextChunks reads a text file and splits it into overlapping chunks.
func loadTextChunks(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := normalizeLineBreaks(string(data))
	return splitTextIntoChunks(text, defTextChunkSize, defTextChunkOverlap), nil
}

// loadCSVChunks reads a CSV file and groups rows into text chunks.
func loadCSVChunks(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	reader.FieldsPerRecord = -1

	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}

	headers := rows[0]
	useHeader := hasLikelyHeader(headers) && len(rows) > 1

	startRow := 0
	if useHeader {
		startRow = 1
	}

	lines := make([]string, 0, len(rows)-startRow)
	for rowIdx := startRow; rowIdx < len(rows); rowIdx++ {
		if line := stringifyCSVRow(rowIdx+1, rows[rowIdx], headers, useHeader); line != "" {
			lines = append(lines, line)
		}
	}

	return groupLinesIntoChunks(lines, defCSVRowsPerChunk), nil
}

// splitTextIntoChunks splits text into overlapping rune-based chunks.
func splitTextIntoChunks(text string, chunkSize, overlap int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if chunkSize <= 0 {
		chunkSize = defTextChunkSize
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= chunkSize {
		overlap = chunkSize / 5
	}

	runes := []rune(text)
	if len(runes) <= chunkSize {
		return []string{text}
	}

	step := chunkSize - overlap
	if step <= 0 {
		step = chunkSize
	}

	chunks := make([]string, 0, (len(runes)+step-1)/step)
	for start := 0; start < len(runes); start += step {
		end := start + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		if chunk := strings.TrimSpace(string(runes[start:end])); chunk != "" {
			chunks = append(chunks, chunk)
		}
		if end == len(runes) {
			break
		}
	}
	return chunks
}

// groupLinesIntoChunks groups lines into chunks of at most chunkLines lines.
func groupLinesIntoChunks(lines []string, chunkLines int) []string {
	if chunkLines <= 0 {
		chunkLines = defCSVRowsPerChunk
	}
	chunks := make([]string, 0, (len(lines)+chunkLines-1)/chunkLines)
	for start := 0; start < len(lines); start += chunkLines {
		end := start + chunkLines
		if end > len(lines) {
			end = len(lines)
		}
		if chunk := strings.TrimSpace(strings.Join(lines[start:end], "\n")); chunk != "" {
			chunks = append(chunks, chunk)
		}
	}
	return chunks
}

// normalizeLineBreaks converts all line endings to Unix-style \n.
func normalizeLineBreaks(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}

// hasLikelyHeader reports whether the given row looks like a CSV header (no digits).
func hasLikelyHeader(header []string) bool {
	nonEmpty := 0
	for _, v := range header {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if strings.ContainsAny(v, "0123456789") {
			return false
		}
		nonEmpty++
	}
	return nonEmpty > 0
}

// stringifyCSVRow formats a CSV row as a readable string.
func stringifyCSVRow(rowNumber int, row, headers []string, useHeader bool) string {
	parts := make([]string, 0, len(row))
	for colIdx, raw := range row {
		value := strings.TrimSpace(raw)
		if value == "" {
			continue
		}
		if useHeader && colIdx < len(headers) {
			if h := strings.TrimSpace(headers[colIdx]); h != "" {
				parts = append(parts, h+": "+value)
				continue
			}
		}
		parts = append(parts, "column "+strconv.Itoa(colIdx+1)+": "+value)
	}
	if len(parts) == 0 {
		return ""
	}
	return "row " + strconv.Itoa(rowNumber) + ": " + strings.Join(parts, "; ")
}

// buildChunkID constructs a unique chunk ID from its components.
func buildChunkID(baseID, fileIDPart string, fileIdx, chunkIdx, totalChunks int) string {
	if fileIDPart == "" {
		fileIDPart = "file"
	}
	id := fmt.Sprintf("%s_f%03d_%s", baseID, fileIdx+1, fileIDPart)
	if totalChunks > 1 {
		id += fmt.Sprintf("_%03d", chunkIdx+1)
	}
	return id
}

// sanitizeIDPart converts a file path into a safe lowercase identifier segment.
func sanitizeIDPart(filePath string) string {
	name := strings.ToLower(filepath.ToSlash(strings.TrimSpace(filePath)))
	name = strings.TrimSuffix(name, filepath.Ext(name))

	var b strings.Builder
	lastUnderscore := false
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			_, _ = b.WriteRune(r)
			lastUnderscore = false
		} else if !lastUnderscore {
			_ = b.WriteByte('_')
			lastUnderscore = true
		}
	}

	if out := strings.Trim(b.String(), "_"); out != "" {
		return out
	}
	return "file"
}

// resolveKnowledgePath resolves a file reference relative to the base directory
// or working directory, returning an absolute path.
func resolveKnowledgePath(baseDir, fileRef string) string {
	fileRef = filepath.Clean(strings.TrimSpace(fileRef))
	if filepath.IsAbs(fileRef) {
		return fileRef
	}
	if p := filepath.Join(baseDir, fileRef); fileExists(p) {
		return p
	}
	if wd, err := os.Getwd(); err == nil {
		if p := filepath.Join(wd, fileRef); fileExists(p) {
			return p
		}
	}
	return filepath.Join(baseDir, fileRef)
}

// relativePathOrOriginal returns path relative to baseDir, or the original slash-normalized path.
func relativePathOrOriginal(baseDir, targetPath string) string {
	rel, err := filepath.Rel(baseDir, targetPath)
	if err != nil {
		return filepath.ToSlash(targetPath)
	}
	return filepath.ToSlash(rel)
}

// isSupportedExtension reports whether the file has an indexable extension.
func isSupportedExtension(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".csv", ".txt", ".md", ".markdown":
		return true
	default:
		return false
	}
}

// fileExists reports whether path points to an existing file.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// normalizeKnowledgeRole returns roleSystem when role matches "system"
// (case-insensitive), otherwise defKnowledgeRole.
func normalizeKnowledgeRole(role string) string {
	if strings.EqualFold(strings.TrimSpace(role), roleSystem) {
		return roleSystem
	}
	return defKnowledgeRole
}
