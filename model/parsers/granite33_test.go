package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestGranite33Parser(t *testing.T) {
	tests := []struct {
		name              string
		input             string
		expectedContent   string
		expectedThinking  string
		expectedToolCalls []api.ToolCall
		thinkingEnabled   bool
		tools             []api.Tool
		lastMessage       *api.Message
	}{
		{
			name:            "simple_content_no_thinking",
			input:           "This is a simple response.",
			expectedContent: "This is a simple response.",
		},
		{
			name:             "thinking_with_response",
			input:            "<think>Let me reason about this carefully.</think>\n<response>The answer is 42.</response>",
			expectedContent:  "The answer is 42.",
			expectedThinking: "Let me reason about this carefully.",
			thinkingEnabled:  true,
		},
		{
			name:             "thinking_without_response_tags",
			input:            "<think>Some reasoning here.</think>Direct answer.",
			expectedContent:  "Direct answer.",
			expectedThinking: "Some reasoning here.",
			thinkingEnabled:  true,
		},
		{
			name:             "thinking_multiline",
			input:            "<think>Line 1\nLine 2\nLine 3</think>\n<response>Final answer.</response>",
			expectedContent:  "Final answer.",
			expectedThinking: "Line 1\nLine 2\nLine 3",
			thinkingEnabled:  true,
		},
		{
			name: "tool_call_single",
			input: `<|tool_call|>[{"name": "get_weather", "arguments": {"location": "Paris"}}]`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
			},
		},
		{
			name: "tool_call_multiple",
			input: `<|tool_call|>[{"name": "get_weather", "arguments": {"location": "Paris"}}, {"name": "get_weather", "arguments": {"location": "London"}}]`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "London",
						}),
					},
				},
			},
		},
		{
			name:            "tool_call_single_object",
			input:           `<|tool_call|>{"name": "get_time", "arguments": {}}`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_time",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name:            "content_before_tool_call",
			input:           `Let me check the weather.<|tool_call|>[{"name": "get_weather", "arguments": {"location": "SF"}}]`,
			expectedContent: "Let me check the weather.",
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "SF",
						}),
					},
				},
			},
		},
		{
			name:             "thinking_disabled_with_think_tags",
			input:            "Content with <think>tags</think> treated as content.",
			expectedContent:  "Content with <think>tags</think> treated as content.",
			expectedThinking: "",
			thinkingEnabled:  false,
		},
		{
			name:             "thinking_disabled_when_tools_present",
			input:            `<|tool_call|>[{"name": "get_weather", "arguments": {"location": "NY"}}]`,
			expectedThinking: "",
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "NY",
						}),
					},
				},
			},
			thinkingEnabled: true,
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
					},
				},
			},
		},
		{
			name:            "prefill_disables_thinking",
			input:           "Continuing from where I left off.",
			expectedContent: "Continuing from where I left off.",
			thinkingEnabled: true,
			lastMessage: &api.Message{
				Role:    "assistant",
				Content: "Previous content",
			},
		},
		{
			name:            "response_tags_stripped",
			input:           "<response>Just an answer.</response>",
			expectedContent: "Just an answer.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Granite33Parser{}
			parser.Init(tt.tools, tt.lastMessage, &api.ThinkValue{Value: tt.thinkingEnabled})

			content, thinking, toolCalls, err := parser.Add(tt.input, true)
			if err != nil {
				t.Fatalf("Add() error = %v", err)
			}

			if diff := cmp.Diff(tt.expectedContent, content); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedThinking, thinking); diff != "" {
				t.Errorf("thinking mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedToolCalls, toolCalls, argsComparer); diff != "" {
				t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGranite33Parser_Streaming(t *testing.T) {
	parser := &Granite33Parser{}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	chunks := []string{
		"<think>",
		"Let me think ",
		"about this.",
		"</think>\n<respon",
		"se>The answer is 42.",
		"</response>",
	}

	var finalContent, finalThinking strings.Builder
	var finalToolCalls []api.ToolCall

	for i, chunk := range chunks {
		done := i == len(chunks)-1
		content, thinking, toolCalls, err := parser.Add(chunk, done)
		if err != nil {
			t.Fatalf("Add() error on chunk %d: %v", i, err)
		}

		finalContent.WriteString(content)
		finalThinking.WriteString(thinking)
		finalToolCalls = append(finalToolCalls, toolCalls...)
	}

	expectedThinking := "Let me think about this."
	expectedContent := "The answer is 42."

	if finalThinking.String() != expectedThinking {
		t.Errorf("expected thinking %q, got %q", expectedThinking, finalThinking.String())
	}

	if finalContent.String() != expectedContent {
		t.Errorf("expected content %q, got %q", expectedContent, finalContent.String())
	}

	if len(finalToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(finalToolCalls))
	}
}

func TestGranite33Parser_StreamingToolCall(t *testing.T) {
	parser := &Granite33Parser{}
	parser.Init(nil, nil, nil)

	chunks := []string{
		`<|tool_call|>[{"name": "get_`,
		`weather", "arguments": {"location":`,
		` "Paris"}}]`,
	}

	var finalContent, finalThinking strings.Builder
	var finalToolCalls []api.ToolCall

	for i, chunk := range chunks {
		done := i == len(chunks)-1
		content, thinking, toolCalls, err := parser.Add(chunk, done)
		if err != nil {
			t.Fatalf("Add() error on chunk %d: %v", i, err)
		}

		finalContent.WriteString(content)
		finalThinking.WriteString(thinking)
		finalToolCalls = append(finalToolCalls, toolCalls...)
	}

	expectedToolCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name: "get_weather",
				Arguments: testArgs(map[string]any{
					"location": "Paris",
				}),
			},
		},
	}

	if finalContent.String() != "" {
		t.Errorf("expected no content, got %q", finalContent.String())
	}

	if diff := cmp.Diff(expectedToolCalls, finalToolCalls, argsComparer); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestGranite33Parser_StreamingEdgeCases(t *testing.T) {
	tests := []struct {
		name              string
		chunks            []string
		expectedContent   string
		expectedThinking  string
		expectedToolCalls []api.ToolCall
		thinkingEnabled   bool
	}{
		{
			name: "split_think_close_tag",
			chunks: []string{
				"Thinking here</thi",
				"nk>Content after.",
			},
			expectedContent:  "Content after.",
			expectedThinking: "Thinking here",
			thinkingEnabled:  true,
		},
		{
			name: "split_tool_call_tag",
			chunks: []string{
				"Before<|tool_ca",
				`ll|>[{"name": "fn", "arguments": {}}]`,
			},
			expectedContent: "Before",
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "fn",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
			thinkingEnabled: false,
		},
		{
			name: "no_thinking_passthrough",
			chunks: []string{
				"Hello ",
				"world!",
			},
			expectedContent:  "Hello world!",
			expectedThinking: "",
			thinkingEnabled:  false,
		},
		{
			name: "split_response_tags",
			chunks: []string{
				"<respon",
				"se>Content here</res",
				"ponse>",
			},
			expectedContent:  "Content here",
			expectedThinking: "",
			thinkingEnabled:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Granite33Parser{}
			parser.Init(nil, nil, &api.ThinkValue{Value: tt.thinkingEnabled})

			var finalContent, finalThinking strings.Builder
			var finalToolCalls []api.ToolCall

			for i, chunk := range tt.chunks {
				done := i == len(tt.chunks)-1
				content, thinking, toolCalls, err := parser.Add(chunk, done)
				if err != nil {
					t.Fatalf("Add() error on chunk %d: %v", i, err)
				}

				finalContent.WriteString(content)
				finalThinking.WriteString(thinking)
				finalToolCalls = append(finalToolCalls, toolCalls...)
			}

			if finalContent.String() != tt.expectedContent {
				t.Errorf("expected content %q, got %q", tt.expectedContent, finalContent.String())
			}

			if finalThinking.String() != tt.expectedThinking {
				t.Errorf("expected thinking %q, got %q", tt.expectedThinking, finalThinking.String())
			}

			if diff := cmp.Diff(tt.expectedToolCalls, finalToolCalls, argsComparer); diff != "" {
				t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGranite33Parser_HasToolSupport(t *testing.T) {
	parser := &Granite33Parser{}
	if !parser.HasToolSupport() {
		t.Error("Granite33Parser should support tools")
	}
}

func TestGranite33Parser_HasThinkingSupport(t *testing.T) {
	parser := &Granite33Parser{}
	if !parser.HasThinkingSupport() {
		t.Error("Granite33Parser should support thinking")
	}
}

func TestGranite33Parser_Init(t *testing.T) {
	t.Run("thinking_enabled_no_tools", func(t *testing.T) {
		parser := &Granite33Parser{}
		tools := parser.Init(nil, nil, &api.ThinkValue{Value: true})
		if parser.state != granite33CollectingThinking {
			t.Errorf("expected state CollectingThinking, got %d", parser.state)
		}
		if tools != nil {
			t.Errorf("expected nil tools, got %v", tools)
		}
	})

	t.Run("thinking_enabled_with_tools", func(t *testing.T) {
		parser := &Granite33Parser{}
		inputTools := []api.Tool{{Function: api.ToolFunction{Name: "test"}}}
		parser.Init(inputTools, nil, &api.ThinkValue{Value: true})
		if parser.state != granite33CollectingContent {
			t.Errorf("expected state CollectingContent (tools disable thinking), got %d", parser.state)
		}
	})

	t.Run("thinking_disabled", func(t *testing.T) {
		parser := &Granite33Parser{}
		parser.Init(nil, nil, &api.ThinkValue{Value: false})
		if parser.state != granite33CollectingContent {
			t.Errorf("expected state CollectingContent, got %d", parser.state)
		}
	})

	t.Run("thinking_nil", func(t *testing.T) {
		parser := &Granite33Parser{}
		parser.Init(nil, nil, nil)
		if parser.state != granite33CollectingContent {
			t.Errorf("expected state CollectingContent, got %d", parser.state)
		}
	})

	t.Run("prefill_disables_thinking", func(t *testing.T) {
		parser := &Granite33Parser{}
		parser.Init(nil, &api.Message{Role: "assistant", Content: "existing"}, &api.ThinkValue{Value: true})
		if parser.state != granite33CollectingContent {
			t.Errorf("expected state CollectingContent (prefill disables thinking), got %d", parser.state)
		}
	})
}
