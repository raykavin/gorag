package cache

import (
	"testing"
	"time"

	"github.com/raykavin/gorag/pkg/models"
)

func TestNormalizeText(t *testing.T) {
	got := normalizeText("  Hello,   WORLD!!!\nHow are-you?  ")
	want := "hello world how are you"
	if got != want {
		t.Fatalf("normalizeText() = %q, want %q", got, want)
	}
}

func TestBuildResponseCacheKeyIgnoresCasePunctuation(t *testing.T) {
	k1, n1 := buildResponseCacheKey("Hello, World!", "User: Hi")
	k2, n2 := buildResponseCacheKey(" hello world ", "user hi")

	if n1 != n2 {
		t.Fatalf("normalized keys differ: %q vs %q", n1, n2)
	}
	if k1 != k2 {
		t.Fatalf("hash keys differ: %q vs %q", k1, k2)
	}
}

func TestCloneChatResponseDeepCopy(t *testing.T) {
	errText := "boom"
	in := &models.ChatResponse{
		Output: "ok",
		ToolCalls: []models.ToolCallLog{{
			Name:     "weather",
			Duration: time.Second,
		}},
		Error: &errText,
	}

	out := cloneChatResponse(in)
	if out == in {
		t.Fatal("expected cloned pointer")
	}
	if &out.ToolCalls[0] == &in.ToolCalls[0] {
		t.Fatal("tool calls slice was not copied")
	}
	if out.Error == in.Error {
		t.Fatal("error pointer was not copied")
	}

	out.ToolCalls[0].Name = "changed"
	*out.Error = "changed"

	if in.ToolCalls[0].Name != "weather" {
		t.Fatalf("input tool call mutated: %q", in.ToolCalls[0].Name)
	}
	if *in.Error != "boom" {
		t.Fatalf("input error mutated: %q", *in.Error)
	}
}
