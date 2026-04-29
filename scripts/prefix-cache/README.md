# Prefix KV Cache Test Scripts

Python scripts to validate and benchmark the prefix KV cache feature (`cache_prompt` / `cached_tokens`).

**No external dependencies** — only Python 3.10+ stdlib.

## Scripts

### `test_prefix_cache_fix.py` — Automated Test Battery (7 tests)

Validates the fix end-to-end by running against both stock and patched Ollama:

| Test | What it checks |
|------|----------------|
| **T0** | Stock Ollama does NOT return `cached_tokens` (baseline) |
| **T1** | Patched server always includes `cached_tokens` in responses |
| **T2** | Second turn in a conversation reuses prefix (~96% cache hit) |
| **T3** | Measurable TTFT reduction with cache vs without (~89% faster) |
| **T4** | `cache_prompt: false` correctly disables caching (`cached_tokens=0`) |
| **T5** | Different system prompt causes cache miss (only BOS tokens match) |
| **T6** | Cache hits increase monotonically across 3 conversation turns |

#### Quick start

```bash
# 1. Build patched Ollama
go build -o ollama-patched .

# 2. Start patched server on a separate port
OLLAMA_HOST=127.0.0.1:11435 ./ollama-patched serve

# 3. Run tests (stock Ollama should already be running on :11434)
python scripts/prefix-cache/test_prefix_cache_fix.py --model granite4:micro
```

#### Options

```
--model MODEL         Model name (default: granite4:micro)
--patched-only        Skip stock Ollama baseline tests
--stock-only          Only run stock Ollama baseline
--patched-url URL     Patched server URL (default: http://127.0.0.1:11435)
--stock-url URL       Stock server URL (default: http://127.0.0.1:11434)
```

#### Example output

```
  [PASS] T3: TTFT reduction when prefix is cached vs uncached
         TTFT reduced by 89% (uncached=0.487s → cached=0.056s), cache: 0 → 430 tokens
         cached:   TTFT=0.056s | prompt=431 tok | cached=430 (100%) | prompt_eval=12ms
         uncached: TTFT=0.487s | prompt=431 tok | cached=0          | prompt_eval=429ms

  7/7 tests passed
```

### `benchmark_prefix_cache.py` — Quick Benchmark

Simulates a 3-turn conversation and prints TTFT + cache metrics per turn:

```bash
python scripts/prefix-cache/benchmark_prefix_cache.py --model granite4:micro

# Or against a specific server
python scripts/prefix-cache/benchmark_prefix_cache.py --url http://127.0.0.1:11435
```
