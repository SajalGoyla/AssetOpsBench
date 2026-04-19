# MCP Workflow Optimization — Design & Implementation

## Overview

This document describes the two MCP-side optimizations implemented for the AssetOpsBench plan-execute agent pipeline. Both optimizations target different phases of the query workflow to reduce end-to-end latency:

| Optimization | Target Phase | Technique | Expected Improvement |
|---|---|---|---|
| Discovery Phase Caching | Phase 1 (Discovery) | Disk-backed cache with smart invalidation | ~2.3s → ~0.005s per query |
| Parallel Step Execution | Phase 3 (Execution) | DAG-based concurrent step dispatch | Wall time collapses for independent steps |

All optimization code is consolidated in a single file (`timer.py`) which wraps the existing pipeline components without modifying them, preserving the original `plan-execute` CLI as an untouched baseline.

---

## Background: The 4-Phase Query Workflow

Every query in AssetOpsBench goes through four sequential phases:

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 1: DISCOVERY (~2.3s)          │
│  Spawn 6 MCP servers via stdio,     │
│  call list_tools() on each,          │
│  collect tool signatures             │
└──────────────┬───────────────────────┘
               │ dict[server_name → tool catalog]
               ▼
┌──────────────────────────────────────┐
│  Phase 2: PLANNING (~3-5s)           │
│  LLM generates a multi-step plan    │
│  with tool names and dependencies    │
└──────────────┬───────────────────────┘
               │ Plan(steps=[PlanStep, ...])
               ▼
┌──────────────────────────────────────┐
│  Phase 3: EXECUTION (varies)         │
│  For each step:                      │
│    - LLM resolves tool arguments     │
│    - MCP call_tool() executes it     │
│  Steps run strictly sequentially     │
└──────────────┬───────────────────────┘
               │ list[StepResult]
               ▼
┌──────────────────────────────────────┐
│  Phase 4: SUMMARIZATION (~0.5-1s)    │
│  LLM combines all step results      │
│  into a final user-facing answer     │
└──────────────────────────────────────┘
```

The baseline path (`uv run plan-execute "query"`) runs all four phases with no caching and strict sequential execution. The optimizations target Phases 1 and 3.

---

## Optimization 1: Discovery Phase Caching

### Problem

Phase 1 spawns all 6 MCP servers (`iot`, `utilities`, `fmsr`, `tsfm`, `wo`, `vibration`) as stdio subprocesses, initializes MCP sessions, and calls `list_tools()` on each — just to read tool signatures that almost never change. This costs **~2.3 seconds per query**, even though the tool catalog only changes when server code is edited or agents are added/removed.

### Solution

A `DiscoveryCache` class in `timer.py` persists the tool signatures to disk (`.discovery_cache.json`) and serves them on subsequent queries, bypassing server spawning entirely.

**How it works:**

```
Query arrives
    │
    ▼
DiscoveryCache.load()
    │
    ├── Cache file exists?
    │     ├── No  → fall through to live discovery
    │     └── Yes → check validity:
    │           ├── MD5 key matches? (server names + file mtimes + pyproject.toml)
    │           │     ├── No  → delete stale cache, print reason, fall through
    │           │     └── Yes → check TTL (24h)
    │           │           ├── Expired → delete, fall through
    │           │           └── Valid   → return cached descriptions (~0.005s)
    │
    ▼ (cache miss)
Executor.get_server_descriptions()  → spawns servers, collects tools (~2.3s)
    │
    ▼
DiscoveryCache.save()  → write to .discovery_cache.json
```

### Cache Invalidation Strategy

The cache key is an MD5 hash incorporating three layers of change detection:

| Trigger | What's tracked | Detects |
|---|---|---|
| Agent addition/removal | Sorted list of server names from `DEFAULT_SERVER_PATHS` | New MCP server registered or old one removed |
| Agent code change | `os.path.getmtime()` of every `.py` file recursively under `src/servers/<name>/` | Any edit to server code — main module, helpers, data loaders |
| Repo config change | `os.path.getmtime()` of `pyproject.toml` | New entry-point registrations, dependency changes |
| TTL expiry | 24-hour default | Safety net for changes not covered above |

When the cache is invalidated, a diagnostic message tells the user *why*:

```
Cache invalidated: new agent(s) added: vibration
Cache invalidated: agent code or repo config changed (file edits detected)
Cache invalidated: TTL expired (25.3h > 24h)
```

### Cache file format (`.discovery_cache.json`)

```json
{
  "key": "85369ce4a6dbe6eaafd36ad5080cb1ce",
  "timestamp": 1776641159.72,
  "server_names": ["fmsr", "iot", "tsfm", "utilities", "vibration", "wo"],
  "descriptions": {
    "iot": "  - sites(): Retrieves a list of sites...",
    "utilities": "  - json_reader(file_name: string): ...",
    ...
  }
}
```

### Result

| Metric | Without Cache | With Cache |
|---|---|---|
| Discovery time | ~2.3s | ~0.005s |
| Speedup | — | ~460x |

---

## Optimization 2: Parallel Step Execution

### Problem

The baseline executor runs plan steps in strict sequential order, even when steps have no data dependencies on each other. For a query like *"Get current time, list assets at site MAIN, and get failure modes for chiller"*, three independent tool calls wait for each other unnecessarily.

### Solution

Steps are grouped into **dependency layers** using Kahn's topological layering algorithm (implemented in `models.py` as `Plan.dependency_layers()`). Steps within the same layer have no dependencies on each other and are dispatched concurrently via `asyncio.gather()`. The next layer starts only after the current layer completes.

**How it works:**

Given a plan with dependencies:

```
Step 1: Get current time          (no deps)
Step 2: List assets at MAIN       (no deps)
Step 3: Get sensors for Chiller   (depends on Step 2)
Step 4: Get failure modes          (depends on Step 2)
```

The `dependency_layers()` method produces:

```
Layer 0: [Step 1, Step 2]    ← run simultaneously via asyncio.gather()
Layer 1: [Step 3, Step 4]    ← run simultaneously after Layer 0 completes
```

**Sequential execution** (baseline):

```
Step 1 ──────► Step 2 ──────► Step 3 ──────► Step 4
         1s          1s            1s            1s    = 4s total
```

**Parallel execution** (optimized):

```
Layer 0:  Step 1 ─────►
          Step 2 ─────►  (wall time = max(Step1, Step2) = 1s)
                   │
Layer 1:  Step 3 ─────►
          Step 4 ─────►  (wall time = max(Step3, Step4) = 1s)

Total wall time: 1s + 1s = 2s  (vs 4s sequential → 2x speedup)
```

### When parallelism helps vs. doesn't

- **Helps:** Multi-step queries with independent sub-tasks (e.g., "get time AND list assets AND get failure modes")
- **Doesn't help:** Simple queries where every step depends on the previous one (e.g., "What assets are at site MAIN?" → Step 1: get sites, Step 2: get assets at that site). These form a chain with no parallelism opportunity.

### Kahn's Algorithm (`Plan.dependency_layers()`)

Located in `src/agent/plan_execute/models.py` (not modified — already existed):

1. Build an in-degree map and a dependents adjacency list from `PlanStep.dependencies`
2. Collect all steps with `in_degree == 0` → they form Layer 0
3. For each step in the current layer, decrement the in-degree of its dependents
4. Any dependent whose in-degree reaches 0 goes into the next layer
5. Repeat until all steps are layered

---

## Files Involved

### Modified Files

| File | What Changed |
|---|---|
| `timer.py` | Added `DiscoveryCache` class (cache + invalidation). Added parallel execution via `dependency_layers()` + `asyncio.gather()`. Added `ProfiledRunner` that wraps the existing pipeline with timing instrumentation. Added `--optimize` benchmark mode, `--cache-discovery` flag, `--clear-cache` command, retry logic for transient errors, and median-based statistical reporting. |

### Existing Files Used (not modified)

| File | What's Imported / Used |
|---|---|
| `src/agent/plan_execute/executor.py` | `Executor` (for `get_server_descriptions()`), `DEFAULT_SERVER_PATHS` (the 6 MCP server registry), `_resolve_args_with_llm()` (LLM-based tool arg resolution), `_call_tool()` (MCP tool invocation), `_list_tools()` (MCP tool listing) |
| `src/agent/plan_execute/planner.py` | `Planner.generate_plan()` — takes question + server descriptions, returns a `Plan` via LLM |
| `src/agent/plan_execute/models.py` | `Plan` (with `dependency_layers()` for parallel and `resolved_order()` for sequential), `PlanStep`, `StepResult` |
| `src/agent/plan_execute/runner.py` | `_SUMMARIZE_PROMPT` — the prompt template for Phase 4 summarization |
| `src/agent/plan_execute/executor_parallel.py` | `ParallelExecutor` — subclass used by the `plan-execute` CLI when `--parallel` is passed (timer.py reimplements this logic with timing) |
| `src/llm/litellm.py` | `LiteLLMBackend` — handles all LLM calls to WatsonX |
| `src/llm/base.py` | `LLMBackend` — abstract interface |

### Design Decision: Why one file?

All optimization logic lives in `timer.py` rather than modifying the existing pipeline files. This is intentional:

- The original `uv run plan-execute "query"` path remains completely untouched, serving as a clean baseline for A/B comparison
- Both optimizations and their profiling instrumentation are co-located for easy review
- The cache, parallel dispatch, and timing code are tightly coupled — separating them would add complexity without benefit

---

## How to Use

### Optimization Benchmark (recommended)

Compares baseline (sequential, no cache) vs optimized (parallel + cache) across multiple runs with median-based statistics:

```bash
uv run python timer.py --optimize "What assets are at site MAIN?"
uv run python timer.py --optimize --runs 7 "Your multi-step query here"
```

### Individual Modes

```bash
uv run python timer.py "query"                          # baseline sequential
uv run python timer.py --cache-discovery "query"        # with discovery cache
uv run python timer.py --parallel "query"               # parallel execution
uv run python timer.py --cache-discovery --parallel "q" # both optimizations
uv run python timer.py --compare "query"                # side-by-side seq vs par
```

### Cache Management

```bash
uv run python timer.py --clear-cache   # manually delete the discovery cache
```

The cache auto-invalidates when server code changes, agents are added/removed, or `pyproject.toml` is edited. Manual clearing is rarely needed.

---

## Statistical Methodology

WatsonX LLM responses have high latency variance (the same call can take 0.5s or 40s depending on server load). To handle this:

- **Median** is used for all comparison metrics (resistant to outliers)
- **5 runs per mode** by default (with 5 samples, median remains valid even if 2 runs are outliers)
- **Outlier detection** flags individual runs where any LLM call exceeds 10s
- **Retry with backoff** on transient WatsonX 500 errors (up to 3 attempts per run)
- **Failed runs are skipped**, not crashed — the benchmark continues with remaining runs

---

## Key Classes and Functions

| Component | Location | Purpose |
|---|---|---|
| `DiscoveryCache` | `timer.py` | Disk-backed cache with MD5 key invalidation + TTL |
| `DiscoveryCache._compute_key()` | `timer.py` | MD5 hash of server names + all `.py` file mtimes + `pyproject.toml` mtime |
| `DiscoveryCache.load()` / `.save()` | `timer.py` | Read/write `.discovery_cache.json` with validation |
| `ProfiledRunner` | `timer.py` | Wraps `Executor`, `Planner`, `LiteLLMBackend` with timing instrumentation |
| `ProfiledRunner.run()` | `timer.py` | Full 4-phase pipeline with optional cache + parallel flags |
| `run_optimization()` | `timer.py` | Benchmark orchestrator: baseline runs → optimized runs → comparison |
| `_run_with_retry()` | `timer.py` | Retry wrapper for transient WatsonX errors |
| `Plan.dependency_layers()` | `models.py` | Kahn's algorithm — groups steps into parallelizable layers |
| `Plan.resolved_order()` | `models.py` | Topological sort for sequential execution |
