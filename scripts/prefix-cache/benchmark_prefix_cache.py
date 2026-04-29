"""
Benchmark: Prefix KV Cache Performance
=======================================
Sends sequential /api/chat requests with shared prefixes and measures:
- TTFT (Time To First Token)
- prompt_eval_count vs cached_tokens
- Whether prefix caching is actually reducing re-evaluation

Usage:
    python scripts/prefix-cache/benchmark_prefix_cache.py [--model granite4:micro]

Options:
    --model MODEL   Model name to test (default: granite4:micro)
    --url URL       Ollama server URL (default: http://localhost:11434)

No external dependencies — uses only Python stdlib.
"""

import argparse
import json
import time
from urllib.request import urlopen, Request

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in software engineering. "
    "You provide concise, accurate answers to technical questions. "
    "Always include code examples when relevant. "
    "Focus on best practices and explain trade-offs."
)

# Pad the system prompt to make prefix caching measurable (~500 tokens)
SYSTEM_PROMPT += " " + ("Context: " + "x" * 80 + ". ") * 20


def chat_request(base_url: str, model: str, messages: list[dict], stream: bool = True) -> dict:
    """Send a /api/chat request and return timing metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_predict": 20},  # short output to focus on prompt eval
    }

    req = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    full_response = ""
    final_metrics = {}

    with urlopen(req, timeout=120) as resp:
        for line in resp:
            if not line.strip():
                continue
            chunk = json.loads(line)

            if chunk.get("message", {}).get("content") and first_token_time is None:
                first_token_time = time.perf_counter()

            full_response += chunk.get("message", {}).get("content", "")

            if chunk.get("done"):
                final_metrics = {
                    "total_duration_ms": chunk.get("total_duration", 0) / 1e6,
                    "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                    "prompt_eval_duration_ms": chunk.get("prompt_eval_duration", 0) / 1e6,
                    "eval_count": chunk.get("eval_count", 0),
                    "eval_duration_ms": chunk.get("eval_duration", 0) / 1e6,
                    "cached_tokens": chunk.get("cached_tokens", 0),
                }

    wall_time = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else wall_time

    return {
        "wall_time_s": round(wall_time, 3),
        "ttft_s": round(ttft, 3),
        "response": full_response[:80],
        **final_metrics,
    }


def run_benchmark(base_url: str, model: str):
    print(f"=== Prefix Cache Benchmark ({model}) ===")
    print(f"    Server: {base_url}\n")

    messages_base = [{"role": "system", "content": SYSTEM_PROMPT}]

    # --- Test 1: Cold start (first request, no cache) ---
    print("Test 1: Cold start (no cache)")
    msgs = messages_base + [{"role": "user", "content": "What is a hash map?"}]
    r1 = chat_request(base_url, model, msgs)
    print_result(r1)

    # --- Test 2: Same prefix, new user message (should hit cache) ---
    print("\nTest 2: Same prefix + new user message (expect cache hit)")
    msgs = messages_base + [
        {"role": "user", "content": "What is a hash map?"},
        {"role": "assistant", "content": r1["response"]},
        {"role": "user", "content": "How does it handle collisions?"},
    ]
    r2 = chat_request(base_url, model, msgs)
    print_result(r2)

    # --- Test 3: Third turn (even longer shared prefix) ---
    print("\nTest 3: Third turn (longer shared prefix, expect more cache hits)")
    msgs = messages_base + [
        {"role": "user", "content": "What is a hash map?"},
        {"role": "assistant", "content": r1["response"]},
        {"role": "user", "content": "How does it handle collisions?"},
        {"role": "assistant", "content": r2["response"]},
        {"role": "user", "content": "What about open addressing?"},
    ]
    r3 = chat_request(base_url, model, msgs)
    print_result(r3)

    # --- Test 4: Different system prompt (should NOT hit cache) ---
    print("\nTest 4: Different system prompt (expect cache miss)")
    msgs_diff = [
        {"role": "system", "content": "You are a pirate. Speak like a pirate."},
        {"role": "user", "content": "What is a hash map?"},
    ]
    r4 = chat_request(base_url, model, msgs_diff)
    print_result(r4)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"{'Test':<35} {'TTFT':>8} {'Prompt Tok':>11} {'Cached':>8} {'Cache %':>8}")
    print("-" * 75)
    for label, r in [
        ("1: Cold start", r1),
        ("2: Same prefix + new msg", r2),
        ("3: Third turn", r3),
        ("4: Different system prompt", r4),
    ]:
        prompt_eval = r.get("prompt_eval_count", 0)
        cached = r.get("cached_tokens", 0)
        pct = f"{cached / prompt_eval * 100:.0f}%" if prompt_eval > 0 else "N/A"
        print(f"{label:<35} {r['ttft_s']:>7.3f}s {prompt_eval:>10} {cached:>7} {pct:>8}")


def print_result(r: dict):
    cached = r.get("cached_tokens", 0)
    prompt_eval = r.get("prompt_eval_count", 0)
    cache_pct = f"{cached / prompt_eval * 100:.0f}%" if prompt_eval > 0 else "N/A"
    print(f"  Wall: {r['wall_time_s']:.3f}s | TTFT: {r['ttft_s']:.3f}s")
    print(f"  Prompt: {prompt_eval} tok | Cached: {cached} tok ({cache_pct})")
    print(f"  Prompt eval: {r.get('prompt_eval_duration_ms', 0):.0f}ms")
    print(f"  Response: {r['response']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ollama prefix KV cache")
    parser.add_argument("--model", default="granite4:micro", help="Model name")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    args = parser.parse_args()
    run_benchmark(args.url, args.model)
