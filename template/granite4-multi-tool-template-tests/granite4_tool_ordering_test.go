package granite4_multi_tool_template_tests

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

// TestGranite4TemplateToolOrdering verifies that the granite4-instruct.gotmpl
// template produces tool definitions with alphabetically-sorted JSON keys when
// tools are provided. The template uses pure Go template constructs to manually
// build JSON with the key ordering that matches HuggingFace's apply_chat_template
// output (Python json.dumps with sorted keys).
func TestGranite4TemplateToolOrdering(t *testing.T) {
	// When running from the granite4-multi-tool-template-tests directory,
	// the template file is one level up in the template/ directory.
	templatePath := "../granite4-instruct.gotmpl"
	if _, err := os.Stat(templatePath); err != nil {
		// Try relative to repo root (for CI or different working dirs)
		templatePath = "template/granite4-instruct.gotmpl"
	}
	bts, err := os.ReadFile(templatePath)
	if err != nil {
		t.Fatal(err)
	}

	tmpl, err := template.Parse(string(bts))
	if err != nil {
		t.Fatal(err)
	}

	props := api.NewToolPropertiesMap()
	props.Set("location", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The city and state, e.g. San Francisco, CA",
	})

	var buf bytes.Buffer
	err = tmpl.Execute(&buf, template.Values{
		Messages: []api.Message{
			{Role: "user", Content: "What's the weather in NYC?"},
		},
		Tools: api.Tools{
			{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather for a location",
					Parameters: api.ToolFunctionParameters{
						Type:       "object",
						Required:   []string{"location"},
						Properties: props,
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	got := buf.String()

	// The rendered prompt should contain tool definitions with alphabetical key order
	// Specifically, "properties" must come before "required" in the parameters object
	propsIdx := strings.Index(got, `"properties"`)
	reqIdx := strings.Index(got, `"required"`)
	if propsIdx == -1 || reqIdx == -1 {
		t.Fatalf("expected both 'properties' and 'required' in rendered prompt, got: %s", got)
	}
	if propsIdx > reqIdx {
		t.Errorf("granite4 template: 'properties' should come before 'required' in tool JSON (alphabetical order).\nrendered prompt: %s", got)
	}
}
