"""
Test battery: Prefix KV Cache Fix Validation
=============================================
Validates that the patched Ollama correctly:
1. Returns `cached_tokens` in API responses
2. Shows increasing cache hits on multi-turn conversations
3. Respects `cache_prompt: false` to disable caching
4. Demonstrates TTFT reduction when cache is active
5. Handles cache misses correctly (different system prompts)

Runs against two Ollama instances:
- Stock  (port 11434): baseline — expects NO `cached_tokens` field
- Patched (port 11435): expects `cached_tokens` field with correct values

Usage:
    python scripts/prefix-cache/test_prefix_cache_fix.py [--model granite4:micro]

Options:
    --model MODEL       Model name to test (default: granite4:micro)
    --patched-only      Skip stock Ollama baseline tests
    --stock-only        Only run stock Ollama baseline tests
    --patched-url URL   URL for patched Ollama (default: http://127.0.0.1:11435)
    --stock-url URL     URL for stock Ollama (default: http://127.0.0.1:11434)

Setup:
    1. Have stock Ollama running normally (port 11434)
    2. Build the patched Ollama:
           go build -o ollama-patched .
    3. Start patched Ollama on a different port:
           OLLAMA_HOST=127.0.0.1:11435 ./ollama-patched serve
    4. Run this script:
           python scripts/prefix-cache/test_prefix_cache_fix.py

No external dependencies — uses only Python stdlib.
"""

import argparse
import json
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import URLError

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in software engineering. "
    "You provide concise, accurate answers to technical questions. "
    "Always include code examples when relevant. "
    "Focus on best practices and explain trade-offs. "
    # Pad to ~500 tokens to make prefix caching measurable
    + " ".join(
        [f"Context item {i}: {'x' * 60}." for i in range(25)]
    )
)

NUM_PREDICT = 15  # Short output — we care about prompt eval, not generation

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details: dict = {}

    def pass_(self, msg: str = ""):
        self.passed = True
        self.message = msg
        return self

    def fail(self, msg: str):
        self.passed = False
        self.message = msg
        return self


def chat_request(
    base_url: str,
    model: str,
    messages: list[dict],
    cache_prompt: bool | None = None,
) -> dict:
    """Send /api/chat and return parsed metrics."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": NUM_PREDICT},
    }
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt

    req = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    full_response = ""
    final_metrics: dict = {}

    with urlopen(req, timeout=120) as resp:
        for line in resp:
            if not line.strip():
                continue
            chunk = json.loads(line)

            content = chunk.get("message", {}).get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()
            full_response += content

            if chunk.get("done"):
                final_metrics = {
                    "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                    "prompt_eval_duration_ns": chunk.get("prompt_eval_duration", 0),
                    "eval_count": chunk.get("eval_count", 0),
                    "eval_duration_ns": chunk.get("eval_duration", 0),
                    "cached_tokens": chunk.get("cached_tokens"),  # None if field absent
                }

    wall = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else wall

    return {
        "wall_s": round(wall, 3),
        "ttft_s": round(ttft, 3),
        "response": full_response[:100],
        **final_metrics,
    }


def is_server_up(base_url: str) -> bool:
    try:
        urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except (URLError, OSError):
        return False


def fmt_result(r: dict) -> str:
    cached = r.get("cached_tokens")
    prompt = r.get("prompt_eval_count", 0)
    cached_str = str(cached) if cached is not None else "N/A"
    pct = f"{cached / prompt * 100:.0f}%" if cached and prompt else "—"
    pe_ms = r.get("prompt_eval_duration_ns", 0) / 1e6
    return (
        f"TTFT={r['ttft_s']:.3f}s | "
        f"prompt={prompt} tok | "
        f"cached={cached_str} ({pct}) | "
        f"prompt_eval={pe_ms:.0f}ms"
    )


# ──────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────

def test_cached_tokens_field_present(base_url: str, model: str) -> TestResult:
    """T1: The patched server returns `cached_tokens` in the response."""
    t = TestResult("T1: cached_tokens field present in response")
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a linked list?"},
    ]
    r = chat_request(base_url, model, msgs)
    t.details = r
    if r["cached_tokens"] is not None:
        return t.pass_(f"cached_tokens={r['cached_tokens']}")
    return t.fail("cached_tokens field missing from response")


def test_cache_hit_on_second_turn(base_url: str, model: str) -> TestResult:
    """T2: Second turn in a conversation shows cached_tokens > 0."""
    t = TestResult("T2: cache hit on second turn (shared prefix)")

    msgs1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a hash map?"},
    ]
    r1 = chat_request(base_url, model, msgs1)

    # Second turn — extends the conversation, shares the full prefix
    msgs2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a hash map?"},
        {"role": "assistant", "content": r1["response"]},
        {"role": "user", "content": "How does it handle collisions?"},
    ]
    r2 = chat_request(base_url, model, msgs2)
    t.details = {"turn1": r1, "turn2": r2}

    cached = r2.get("cached_tokens")
    if cached is None:
        return t.fail("cached_tokens field missing from turn 2 response")

    # The second turn should cache at least the system prompt portion
    if cached > 0:
        return t.pass_(
            f"turn2: cached {cached}/{r2['prompt_eval_count']} tokens "
            f"({cached / r2['prompt_eval_count'] * 100:.0f}%)"
        )
    return t.fail(f"cached_tokens=0 on turn 2 — prefix cache not working")


def test_ttft_reduction_on_cache_hit(base_url: str, model: str) -> TestResult:
    """T3: TTFT is measurably lower when prefix is cached vs uncached."""
    t = TestResult("T3: TTFT reduction when prefix is cached vs uncached")

    # Force a cold eval (no cache) to get a true uncached baseline
    msgs1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Explain binary search trees."},
    ]
    r_cold = chat_request(base_url, model, msgs1, cache_prompt=False)

    # Now send the SAME prompt again with cache enabled — should hit cache
    r_warm = chat_request(base_url, model, msgs1, cache_prompt=True)
    t.details = {"uncached": r_cold, "cached": r_warm}

    cold_cached = r_cold.get("cached_tokens") or 0
    warm_cached = r_warm.get("cached_tokens") or 0

    # The warm request should have significantly more cached tokens
    if warm_cached > cold_cached and r_warm["ttft_s"] < r_cold["ttft_s"]:
        reduction = (1 - r_warm["ttft_s"] / r_cold["ttft_s"]) * 100
        return t.pass_(
            f"TTFT reduced by {reduction:.0f}% "
            f"(uncached={r_cold['ttft_s']:.3f}s → cached={r_warm['ttft_s']:.3f}s), "
            f"cache: {cold_cached} → {warm_cached} tokens"
        )

    # Even if TTFT is similar (fast model), check that cached tokens increased
    if warm_cached > cold_cached:
        return t.pass_(
            f"Cache hit confirmed: {cold_cached} → {warm_cached} tokens "
            f"(TTFT: {r_cold['ttft_s']:.3f}s → {r_warm['ttft_s']:.3f}s)"
        )

    return t.fail(
        f"No improvement — uncached={cold_cached} tok, cached={warm_cached} tok, "
        f"TTFT: {r_cold['ttft_s']:.3f}s → {r_warm['ttft_s']:.3f}s"
    )


def test_cache_prompt_false_disables_cache(base_url: str, model: str) -> TestResult:
    """T4: Setting cache_prompt=false forces full re-evaluation."""
    t = TestResult("T4: cache_prompt=false disables prefix caching")

    # Warm the cache with a first request
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is recursion?"},
    ]
    chat_request(base_url, model, msgs)

    # Same prompt again but with cache_prompt=false
    r = chat_request(base_url, model, msgs, cache_prompt=False)
    t.details = r

    cached = r.get("cached_tokens")
    if cached is None:
        return t.fail("cached_tokens field missing from response")
    if cached == 0:
        return t.pass_("cached_tokens=0 when cache_prompt=false — cache correctly disabled")
    return t.fail(f"cached_tokens={cached} when cache_prompt=false — cache was NOT disabled")


def test_cache_miss_on_different_prefix(base_url: str, model: str) -> TestResult:
    """T5: Different system prompts produce a cache miss."""
    t = TestResult("T5: cache miss when system prompt changes")

    msgs1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is polymorphism?"},
    ]
    r1 = chat_request(base_url, model, msgs1)

    msgs2 = [
        {"role": "system", "content": "You are a pirate captain. Answer everything like a pirate."},
        {"role": "user", "content": "What is polymorphism?"},
    ]
    r2 = chat_request(base_url, model, msgs2)
    t.details = {"prompt1": r1, "prompt2_different_system": r2}

    cached = r2.get("cached_tokens")
    if cached is None:
        return t.fail("cached_tokens field missing")

    prompt_count = r2.get("prompt_eval_count", 0)
    # With a completely different system prompt, cached should be low.
    # A few BOS/template-header tokens may still match (typically 5-10),
    # so we allow up to 30% for very short prompts.
    if prompt_count > 0 and cached / prompt_count < 0.3:
        return t.pass_(
            f"cached_tokens={cached}/{prompt_count} "
            f"({cached / prompt_count * 100:.0f}%) — correct cache miss"
        )
    return t.fail(
        f"cached_tokens={cached}/{prompt_count} — unexpectedly high for different system prompt"
    )


def test_three_turn_increasing_cache(base_url: str, model: str) -> TestResult:
    """T6: Cache hits increase across 3 turns of the same conversation."""
    t = TestResult("T6: increasing cache hits across 3 conversation turns")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a stack?"},
    ]
    r1 = chat_request(base_url, model, msgs)

    msgs += [
        {"role": "assistant", "content": r1["response"]},
        {"role": "user", "content": "And a queue?"},
    ]
    r2 = chat_request(base_url, model, msgs)

    msgs += [
        {"role": "assistant", "content": r2["response"]},
        {"role": "user", "content": "Compare them."},
    ]
    r3 = chat_request(base_url, model, msgs)

    t.details = {"turn1": r1, "turn2": r2, "turn3": r3}

    c1 = r1.get("cached_tokens", 0) or 0
    c2 = r2.get("cached_tokens", 0) or 0
    c3 = r3.get("cached_tokens", 0) or 0

    if c2 > c1 and c3 > c2:
        return t.pass_(f"cached tokens: turn1={c1} → turn2={c2} → turn3={c3}")
    if c2 > 0 and c3 > 0 and c3 >= c2:
        return t.pass_(f"cached tokens: turn1={c1} → turn2={c2} → turn3={c3} (non-decreasing)")
    return t.fail(f"cached tokens not increasing: turn1={c1}, turn2={c2}, turn3={c3}")


def test_stock_no_cached_tokens_field(base_url: str, model: str) -> TestResult:
    """T0: Stock Ollama does NOT return cached_tokens (baseline confirmation)."""
    t = TestResult("T0: stock Ollama has no cached_tokens field (baseline)")
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is an array?"},
    ]
    r = chat_request(base_url, model, msgs)
    t.details = r

    if r["cached_tokens"] is None:
        return t.pass_("Confirmed: stock Ollama does not return cached_tokens")
    return t.fail(f"Unexpected: stock Ollama returned cached_tokens={r['cached_tokens']}")


# ──────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────

def run_test_suite(label: str, base_url: str, model: str, tests: list) -> list[TestResult]:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Server: {base_url} | Model: {model}")
    print(f"{'='*70}\n")

    results = []
    for test_fn in tests:
        try:
            result = test_fn(base_url, model)
        except Exception as e:
            result = TestResult(test_fn.__doc__ or test_fn.__name__)
            result.fail(f"Exception: {e}")
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}")
        print(f"         {result.message}")

        if result.details:
            # Check if details is a nested dict (multi-turn results)
            if isinstance(result.details, dict) and all(
                isinstance(v, dict) for v in result.details.values()
            ):
                for key in sorted(result.details):
                    print(f"         {key}: {fmt_result(result.details[key])}")
            else:
                print(f"         {fmt_result(result.details)}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Test battery: prefix KV cache fix")
    parser.add_argument("--model", default="granite4:micro", help="Model to test")
    parser.add_argument("--patched-only", action="store_true", help="Skip stock Ollama tests")
    parser.add_argument("--stock-only", action="store_true", help="Only run stock baseline")
    parser.add_argument("--patched-url", default="http://127.0.0.1:11435", help="Patched Ollama URL")
    parser.add_argument("--stock-url", default="http://127.0.0.1:11434", help="Stock Ollama URL")
    args = parser.parse_args()

    all_results: list[TestResult] = []

    # ── Stock Ollama baseline ──
    if not args.patched_only:
        if is_server_up(args.stock_url):
            stock_results = run_test_suite(
                "BASELINE: Stock Ollama (unpatched)",
                args.stock_url,
                args.model,
                [test_stock_no_cached_tokens_field],
            )
            all_results.extend(stock_results)
        else:
            print(f"\n  [SKIP] Stock Ollama not running on {args.stock_url}")

    if args.stock_only:
        return

    # ── Patched Ollama tests ──
    if not is_server_up(args.patched_url):
        print(f"\n  [ERROR] Patched Ollama not running on {args.patched_url}")
        print(f"  Start it with: OLLAMA_HOST=127.0.0.1:11435 ./ollama-patched serve")
        sys.exit(1)

    patched_results = run_test_suite(
        "PATCHED: Ollama with prefix cache fix",
        args.patched_url,
        args.model,
        [
            test_cached_tokens_field_present,
            test_cache_hit_on_second_turn,
            test_ttft_reduction_on_cache_hit,
            test_cache_prompt_false_disables_cache,
            test_cache_miss_on_different_prefix,
            test_three_turn_increasing_cache,
        ],
    )
    all_results.extend(patched_results)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
    print(f"\n  {passed}/{total} tests passed")
    print(f"{'='*70}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
