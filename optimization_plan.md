# AssetOpsBench — Optimization Plan

This document captures the full optimization roadmap identified during profiling of the
plan-execute workflow. It is intended to bring a new session up to speed on what has
already been done, what remains, and how to approach each step.

---

## Context: The Workflow Being Optimized

Every `plan-execute` query goes through four phases:

```
Discover → Plan → Execute → Summarize
```

Profiler baseline (3-run average, query: "What assets are available at site MAIN?"):

| Phase | Avg time |
|---|---|
| Discovery | 3.398s |
| Planning (LLM) | 3.336s |
| Execution | 1.013s |
| Summarization | 0.931s |
| **Total** | **8.678s** |

The profiler script is at `src/workflow/profiler.py`. Run it with:
```bash
uv run python -m workflow.profiler --runs 3 "Your question"
uv run python -m workflow.profiler --runs 3 --cache-discovery "Your question"
```

The working model is `watsonx/meta-llama/llama-3-3-70b-instruct`.
The default model in the codebase (`llama-4-maverick`) currently returns a 500 from WatsonX.

---

## Optimization 1: Discovery Caching

**File:** `src/workflow/executor_cached.py` (new file, subclasses `Executor`)

**Problem:** Every query spawned all 5 MCP servers just to list their tool signatures,
which never change between queries. Cost: ~3.4s per query.

**Solution:** Cache tool signatures to disk at `src/workflow/.discovery_cache.json`.
Subsequent queries read from the file instead of spawning servers.

**Cache invalidation — three layers:**
1. Server name change — MD5 hash of server names detects additions/removals
2. Source file mtime — modification timestamp of each server's `.py` file is included
   in the hash, so editing server code auto-invalidates the cache
3. TTL — cache expires after 24 hours by default (configurable via `ttl_seconds`)

**How to use:**
```bash
uv run plan-execute --model-id watsonx/meta-llama/llama-3-3-70b-instruct --cache-discovery "question"
```

**Result:** Discovery drops from 3.4s → 0.001s. Total query time: ~8.7s → ~5.5s (-36%).

**Detailed write-up:** `discovery_cache.md`

---

## Optimization 2: Parallel Execution of Independent Steps

**Files to change:** `src/workflow/executor.py` (or a new `executor_parallel.py` subclass)

**Problem:** The executor runs plan steps **sequentially** even when they have no
dependencies on each other. For a 3-step plan where steps 1, 2, 3 are all independent,
they run one after another instead of simultaneously.

Look at `executor.py:90-108` (`execute_plan`):
```python
for step in ordered:
    result = await self.execute_step(step, context, question)  # one at a time
    context[step.step_number] = result
```

**Solution:** Group steps by dependency level (topological layers), then
`asyncio.gather()` all steps in the same layer in parallel. Steps that depend on prior
results still wait, but independent steps run concurrently.

Concretely:
```python
# Instead of sequential loop, group into layers and gather each layer
layers = plan.dependency_layers()   # needs to be implemented on Plan
for layer in layers:
    results = await asyncio.gather(*[
        self.execute_step(step, context, question) for step in layer
    ])
    for r in results:
        context[r.step_number] = r
```

**Expected impact:** For multi-step queries with parallel steps (e.g. the 3-server
parallel example in INSTRUCTIONS.md), wall-clock execution time could drop by up to
N× where N is the number of parallel steps.

**Important note:** The `Plan` model (`src/workflow/models.py`) already has
`resolved_order()` which does topological sorting. It needs a companion
`dependency_layers()` method that groups steps into levels rather than flattening them.

**Approach:** Same subclass pattern used for discovery caching — create
`executor_parallel.py` that subclasses `Executor` and overrides `execute_plan()` only,
keeping original as default.

---

## Optimization 3: Reuse MCP Server Connections Within a Query — NOT STARTED

**Files to change:** `src/workflow/executor.py` (or a new subclass)

**Problem:** Every tool call spawns a fresh subprocess, does one tool call, then kills it.
If a plan has two steps on the same server (e.g. `iot → assets` then `iot → history`),
the `iot` server is spawned and killed twice, paying the full startup cost each time.

Look at `executor.py:328-338` (`_call_tool`):
```python
async def _call_tool(server_path, tool_name, args):
    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):      # spawn
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
    # server killed here
```

**Solution:** Keep MCP sessions alive for the duration of a single query. Open one
session per server at the start of `execute_plan`, reuse it for all steps that target
that server, close all sessions at the end.

```python
# Pseudocode
async with open_sessions(self._server_paths) as sessions:
    for step in ordered:
        session = sessions[step.server]
        result = await session.call_tool(step.tool, step.tool_args)
```

**Expected impact:** For multi-step queries hitting the same server multiple times,
saves ~600ms per repeated server call (subprocess startup cost).

**Complexity:** Medium — requires managing a dict of open sessions across the plan
execution, and ensuring sessions are always cleaned up even on errors.

---

## Optimization 4: Skip Summarization for Single-Step Queries — NOT STARTED

**File to change:** `src/workflow/runner.py`

**Problem:** The summarizer always makes a full LLM call even when the plan has only
one step and the tool result could just be returned directly.

Look at `runner.py:98-106`:
```python
# 4. Summarise
answer = self._llm.generate(
    _SUMMARIZE_PROMPT.format(question=question, results=results_text)
)
```

This always fires regardless of how many steps there are or how simple the result is.

**Solution:** If the plan has exactly one step and it succeeded, return the tool result
directly (possibly with light formatting) instead of making an LLM call.

```python
if len(history) == 1 and history[0].success:
    answer = history[0].response  # or light formatting
else:
    answer = self._llm.generate(_SUMMARIZE_PROMPT.format(...))
```

**Expected impact:** Saves ~0.9s (one LLM call) on all single-step queries, which are
the most common query type for simple lookups.

**Risk:** The raw tool response is JSON, not natural language. For CLI use it may look
ugly. Consider a lightweight template instead of skipping entirely.

---

## Optimization 5: Smarter Placeholder Resolution — NOT STARTED

**File to change:** `src/workflow/executor.py`

**Problem:** For dependent steps (steps whose arguments contain `{step_N}` placeholders),
an extra LLM call is made just to extract a concrete value from the prior step's result.

Look at `executor.py:191-240` (`_resolve_args_with_llm`):
```python
# Called when a step has {step_1} in its args
resolved_args = await _resolve_args_with_llm(
    step.task, step.tool, step.tool_args, context, self._llm
)
```

For a simple case like extracting an asset ID from a JSON response, this burns a full
LLM round-trip (~1-3s) for what is essentially a string extraction.

**Solution:** For simple cases where the prior step returned clean JSON and the
placeholder maps to a top-level key, extract it directly without an LLM call.
Fall back to the LLM only when simple extraction fails.

```python
def _try_simple_resolve(args, context) -> dict | None:
    # Attempt direct JSON key extraction for {step_N} placeholders
    # Return None if the result is ambiguous or nested
    ...
```

**Expected impact:** Saves 1-3s per dependent step on simple extraction cases.

**Complexity:** Medium — need to define "simple" carefully to avoid silently extracting
the wrong value. The LLM fallback must remain reliable.

---

## Recommended Order of Execution

| # | Optimization | Impact | Complexity | Status |
|---|---|---|---|---|
| 1 | Discovery caching | -3.4s (-36%) | Low | **Done** |
| 4 | Skip summarization (single-step) | -0.9s | Low | Not started |
| 2 | Parallel step execution | variable (multi-step) | Medium | Not started |
| 3 | Reuse MCP connections | -0.6s per repeat | Medium | Not started |
| 5 | Smarter placeholder resolution | -1-3s per dep step | Medium | Not started |

Optimization 4 is the next lowest-hanging fruit — it's a 3-line change in `runner.py`
with a clear measurable impact on the most common query type.

---

## Key Files Reference

| File | Purpose |
|---|---|
| `src/workflow/runner.py` | Orchestrates all 4 phases |
| `src/workflow/executor.py` | Executes plan steps, MCP tool calls |
| `src/workflow/executor_cached.py` | Discovery caching subclass (Opt 1) |
| `src/workflow/planner.py` | LLM-based plan generation |
| `src/workflow/models.py` | Plan, PlanStep, StepResult data classes |
| `src/workflow/cli.py` | CLI entry point (`plan-execute` command) |
| `src/workflow/profiler.py` | Phase-level latency profiler |
| `discovery_cache.md` | Deep-dive write-up for Optimization 1 |

---

## Development Conventions Established

- **Never modify original files when possible** — use subclasses instead
  (see `executor_cached.py` pattern)
- **Caching is opt-in** — original behaviour is always the default, new behaviour
  requires an explicit flag (`--cache-discovery`)
- **Always profile before and after** — use `profiler.py` with `--runs 3` for
  stable averages before claiming an improvement
- **Document each optimization** — write a dedicated `.md` explaining what changed,
  why, and what was measured
