package models

// KnowledgeChunk represents a piece of information from the knowledge base.
type KnowledgeChunk struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Category  string    `json:"category"`
	Files     []string  `json:"files,omitempty"`
	FilesPath string    `json:"files_path,omitempty"`
	Source    string    `json:"source,omitempty"`
	Score     float32   `json:"score,omitempty"`
	Embedding []float32 `json:"-"`
}
