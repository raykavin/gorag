package models

import "time"

// ToolCallLog represents a log of a tool execution
type ToolCallLog struct {
	Name     string        `json:"name"`
	Input    string        `json:"input"`
	Output   string        `json:"output"`
	Duration time.Duration `json:"duration_ms"`
	Success  bool          `json:"success"`
}
