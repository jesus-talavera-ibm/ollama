package parsers

import (
	"encoding/json"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type granite33State int

const (
	granite33CollectingThinking granite33State = iota
	granite33CollectingContent
	granite33CollectingToolCalls
)

const (
	granite33ThinkOpenTag    = "<think>"
	granite33ThinkCloseTag   = "</think>"
	granite33ResponseOpenTag = "<response>"
	granite33ResponseCloseTag = "</response>"
	granite33ToolCallTag     = "<|tool_call|>"
)

type Granite33Parser struct {
	state            granite33State
	buffer           strings.Builder
	thinkingEnabled  bool
}

func (p *Granite33Parser) HasToolSupport() bool {
	return true
}

func (p *Granite33Parser) HasThinkingSupport() bool {
	return true
}

func (p *Granite33Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	thinkingEnabled := thinkValue != nil && thinkValue.Bool()
	prefill := lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""

	if thinkingEnabled && !prefill && len(tools) == 0 {
		p.state = granite33CollectingThinking
		p.thinkingEnabled = true
	} else {
		p.state = granite33CollectingContent
		p.thinkingEnabled = false
	}

	return tools
}

type granite33Event interface {
	isGranite33Event()
}

type granite33ThinkingEvent struct {
	content string
}

type granite33ContentEvent struct {
	content string
}

type granite33ToolCallEvent struct {
	toolCall api.ToolCall
}

func (granite33ThinkingEvent) isGranite33Event() {}
func (granite33ContentEvent) isGranite33Event()  {}
func (granite33ToolCallEvent) isGranite33Event() {}

func (p *Granite33Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	if done && p.buffer.Len() > 0 {
		// On final chunk, flush any remaining buffer
		events := p.parseEvents()
		// If there's still something in the buffer after parsing, drain it
		if p.buffer.Len() > 0 {
			remaining := p.buffer.String()
			p.buffer.Reset()
			switch p.state {
			case granite33CollectingThinking:
				events = append(events, granite33ThinkingEvent{content: remaining})
			default:
				// Strip leftover response tags from remaining content
				remaining = stripResponseTags(remaining)
				events = append(events, granite33ContentEvent{content: remaining})
			}
		}
		return collectGranite33Events(events)
	}

	events := p.parseEvents()
	return collectGranite33Events(events)
}

func collectGranite33Events(events []granite33Event) (string, string, []api.ToolCall, error) {
	var contentSb, thinkingSb strings.Builder
	var toolCalls []api.ToolCall

	for _, event := range events {
		switch e := event.(type) {
		case granite33ThinkingEvent:
			thinkingSb.WriteString(e.content)
		case granite33ContentEvent:
			contentSb.WriteString(e.content)
		case granite33ToolCallEvent:
			toolCalls = append(toolCalls, e.toolCall)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Granite33Parser) parseEvents() []granite33Event {
	var all []granite33Event

	keepLooping := true
	for keepLooping {
		var events []granite33Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *Granite33Parser) eat() ([]granite33Event, bool) {
	var events []granite33Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case granite33CollectingThinking:
		// Look for <think> opening tag first (model may emit it)
		if strings.HasPrefix(strings.TrimLeftFunc(bufStr, unicode.IsSpace), granite33ThinkOpenTag) {
			// Strip the <think> tag
			trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace)
			after := trimmed[len(granite33ThinkOpenTag):]
			p.buffer.Reset()
			p.buffer.WriteString(after)
			return events, true
		}

		// Look for </think> closing tag
		if strings.Contains(bufStr, granite33ThinkCloseTag) {
			split := strings.SplitN(bufStr, granite33ThinkCloseTag, 2)
			thinking := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = granite33CollectingContent

			if len(thinking) > 0 {
				events = append(events, granite33ThinkingEvent{content: thinking})
			}
			return events, true
		}

		// Check for partial </think> at the end
		if overlapLen := overlap(bufStr, granite33ThinkCloseTag); overlapLen > 0 {
			beforePartial := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartial)
			ambiguousStart := len(beforePartial) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, granite33ThinkingEvent{content: unambiguous})
			}
			return events, false
		}

		// No closing tag found — emit thinking content but hold back trailing whitespace
		whitespaceLen := trailingWhitespaceLen(bufStr)
		ambiguousStart := len(bufStr) - whitespaceLen

		unambiguous := bufStr[:ambiguousStart]
		ambiguous := bufStr[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, granite33ThinkingEvent{content: unambiguous})
		}
		return events, false

	case granite33CollectingContent:
		// First strip any complete response tags from the buffer
		stripped := stripResponseTags(bufStr)
		if stripped != bufStr {
			p.buffer.Reset()
			p.buffer.WriteString(stripped)
			bufStr = stripped
		}

		// Check for tool call tag
		if strings.Contains(bufStr, granite33ToolCallTag) {
			split := strings.SplitN(bufStr, granite33ToolCallTag, 2)
			contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = granite33CollectingToolCalls

			if len(contentBefore) > 0 {
				events = append(events, granite33ContentEvent{content: contentBefore})
			}
			return events, true
		}

		// Check for partial tags at the end of the buffer — hold them back
		holdBack := 0
		for _, tag := range []string{granite33ToolCallTag, granite33ResponseOpenTag, granite33ResponseCloseTag} {
			if ol := overlap(bufStr, tag); ol > holdBack {
				holdBack = ol
			}
		}

		if holdBack > 0 {
			emittable := bufStr[:len(bufStr)-holdBack]
			p.buffer.Reset()
			p.buffer.WriteString(bufStr[len(bufStr)-holdBack:])
			if len(emittable) > 0 {
				events = append(events, granite33ContentEvent{content: emittable})
			}
			return events, false
		}

		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, granite33ContentEvent{content: bufStr})
		}
		return events, false

	case granite33CollectingToolCalls:
		// Granite 3.3 tool call format: <|tool_call|> followed by JSON array or object
		// e.g.: <|tool_call|>[{"name": "fn", "arguments": {"key": "val"}}]
		// or:   <|tool_call|>{"name": "fn", "arguments": {"key": "val"}}

		// Try to parse the accumulated buffer as JSON
		trimmed := strings.TrimSpace(bufStr)
		if trimmed == "" {
			return events, false
		}

		// Try parsing as JSON array of tool calls
		if strings.HasPrefix(trimmed, "[") {
			toolCalls, err := parseGranite33ToolCallArray(trimmed)
			if err == nil {
				p.buffer.Reset()
				p.state = granite33CollectingContent
				for _, tc := range toolCalls {
					events = append(events, granite33ToolCallEvent{toolCall: tc})
				}
				return events, true
			}
			// Could be incomplete JSON, wait for more
			return events, false
		}

		// Try parsing as single JSON tool call
		if strings.HasPrefix(trimmed, "{") {
			toolCall, err := parseGranite33ToolCall(trimmed)
			if err == nil {
				p.buffer.Reset()
				p.state = granite33CollectingContent
				events = append(events, granite33ToolCallEvent{toolCall: toolCall})
				return events, true
			}
			// Could be incomplete JSON, wait for more
			return events, false
		}

		// Not JSON — treat the whole thing as content and switch back
		p.buffer.Reset()
		p.state = granite33CollectingContent
		content := stripResponseTags(trimmed)
		if len(content) > 0 {
			events = append(events, granite33ContentEvent{content: content})
		}
		return events, false
	}

	return events, false
}

// stripResponseTags removes <response> and </response> tags from content
func stripResponseTags(s string) string {
	s = strings.ReplaceAll(s, granite33ResponseOpenTag, "")
	s = strings.ReplaceAll(s, granite33ResponseCloseTag, "")
	return s
}

type granite33ToolCallJSON struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func parseGranite33ToolCallArray(s string) ([]api.ToolCall, error) {
	var rawCalls []granite33ToolCallJSON
	if err := json.Unmarshal([]byte(s), &rawCalls); err != nil {
		return nil, err
	}

	var toolCalls []api.ToolCall
	for _, raw := range rawCalls {
		tc, err := granite33RawToToolCall(raw)
		if err != nil {
			slog.Warn("granite33 tool call parsing failed", "error", err)
			continue
		}
		toolCalls = append(toolCalls, tc)
	}
	return toolCalls, nil
}

func parseGranite33ToolCall(s string) (api.ToolCall, error) {
	var raw granite33ToolCallJSON
	if err := json.Unmarshal([]byte(s), &raw); err != nil {
		return api.ToolCall{}, err
	}
	return granite33RawToToolCall(raw)
}

func granite33RawToToolCall(raw granite33ToolCallJSON) (api.ToolCall, error) {
	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal(raw.Arguments, &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      raw.Name,
			Arguments: args,
		},
	}, nil
}
