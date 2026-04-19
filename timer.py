"""Integrated MCP Workflow Optimization Profiler.

Contains both MCP-side optimizations in a single file:

  Optimization 1 — Discovery Phase Caching
    Caches MCP tool signatures to disk so Phase 1 (Discovery) drops from
    ~3.4s to ~0.001s on subsequent queries.  Cache invalidates on server
    name changes, source-file edits, or after a configurable TTL (24h).

  Optimization 2 — Parallel Step Execution
    Groups independent plan steps into topological layers and dispatches
    each layer concurrently via asyncio.gather(), collapsing Phase 3
    (Execution) wall time for multi-step plans.

Usage — optimization benchmark (recommended):
    uv run python timer.py --optimize "What assets are at site MAIN?"
    uv run python timer.py --optimize --runs 5 "query"

  Compares Baseline (sequential, no cache) vs Optimized (parallel + cache).
  Retries automatically on transient WatsonX 500 errors.

Usage — individual profiling modes:
    uv run python timer.py "query"                          # baseline
    uv run python timer.py --parallel "query"               # parallel
    uv run python timer.py --cache-discovery "query"        # with cache
    uv run python timer.py --compare "query"                # seq vs par

Usage — cache management:
    uv run python timer.py --clear-cache
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json as _json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_CACHE_PATH = _REPO_ROOT / ".discovery_cache.json"
_DEFAULT_TTL = 86400  # 24 hours


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZATION 1 — Discovery Phase Cache
# ══════════════════════════════════════════════════════════════════════════════


class DiscoveryCache:
    """Disk-backed cache for MCP server tool signatures.

    Invalidation layers (matches whiteboard spec):
      1. New agent addition / removal — MD5 includes sorted server names
      2. Agent code change  — tracks mtime of ALL .py files in each
         server's directory (not just main.py), so editing helpers,
         data loaders, or DSP modules auto-invalidates
      3. Repo-level change  — includes pyproject.toml mtime to catch
         new entry-point registrations or dependency changes
      4. TTL                — 24-hour default expiry as a safety net

    When load() detects a mismatch it deletes the stale cache file and
    prints the invalidation reason so the user sees why a fresh discovery
    is happening.
    """

    def __init__(
        self,
        server_paths: dict[str, Path | str],
        cache_path: Path = _CACHE_PATH,
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        self._server_paths = server_paths
        self._cache_path = cache_path
        self._ttl = ttl

    def _compute_key(self) -> str:
        """MD5 hash combining server names, all source-file mtimes, and pyproject.toml."""
        parts: list[str] = []
        for name in sorted(self._server_paths):
            path = self._server_paths[name]
            server_dir = _REPO_ROOT / "src" / "servers" / name
            file_mtimes: list[str] = []
            if server_dir.is_dir():
                for py_file in sorted(server_dir.rglob("*.py")):
                    try:
                        file_mtimes.append(
                            f"{py_file.relative_to(server_dir)}:{os.path.getmtime(py_file)}"
                        )
                    except OSError:
                        pass
            mtime_str = "|".join(file_mtimes) if file_mtimes else "0"
            parts.append(f"{name}:{path}:{mtime_str}")
        pyproject = _REPO_ROOT / "pyproject.toml"
        if pyproject.exists():
            parts.append(f"pyproject:{os.path.getmtime(pyproject)}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def load(self) -> dict[str, str] | None:
        """Return cached descriptions, or None if cache is invalid / expired.

        On invalidation the stale file is deleted so the next save()
        starts fresh.
        """
        if not self._cache_path.exists():
            return None
        try:
            data = _json.loads(self._cache_path.read_text())
        except (ValueError, OSError):
            self._delete_stale("corrupted cache file")
            return None

        stored_key = data.get("key", "")
        current_key = self._compute_key()
        if stored_key != current_key:
            reason = self._diagnose_invalidation(data)
            self._delete_stale(reason)
            return None

        age = time.time() - data.get("timestamp", 0)
        if age > self._ttl:
            self._delete_stale(f"TTL expired ({age / 3600:.1f}h > {self._ttl / 3600:.0f}h)")
            return None

        return data.get("descriptions")

    def save(self, descriptions: dict[str, str]) -> None:
        """Persist tool signatures to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": self._compute_key(),
            "timestamp": time.time(),
            "server_names": sorted(self._server_paths.keys()),
            "descriptions": descriptions,
        }
        self._cache_path.write_text(_json.dumps(payload, indent=2))

    def clear(self) -> None:
        """Remove the cache file."""
        if self._cache_path.exists():
            self._cache_path.unlink()
            print("  Discovery cache cleared.")
        else:
            print("  No discovery cache to clear.")

    def _delete_stale(self, reason: str) -> None:
        """Delete stale cache and print reason."""
        print(f"  Cache invalidated: {reason}")
        if self._cache_path.exists():
            self._cache_path.unlink()

    def _diagnose_invalidation(self, data: dict) -> str:
        """Compare stored metadata with current state to explain WHY the cache is stale."""
        stored_names = set(data.get("server_names", []))
        current_names = set(self._server_paths.keys())
        added = current_names - stored_names
        removed = stored_names - current_names
        if added:
            return f"new agent(s) added: {', '.join(sorted(added))}"
        if removed:
            return f"agent(s) removed: {', '.join(sorted(removed))}"
        return "agent code or repo config changed (file edits detected)"


# ══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StepTiming:
    step_number: int
    server: str
    task: str
    tool: str
    llm_resolve_s: float = 0.0   # time spent resolving tool args via LLM
    tool_call_s: float = 0.0     # time spent in the MCP tool call
    total_s: float = 0.0
    success: bool = True


@dataclass
class RunTiming:
    question: str
    mode: str = "sequential"          # "sequential" or "parallel"
    cache_discovery: bool = False
    discovery_s: float = 0.0
    planning_s: float = 0.0
    steps: list[StepTiming] = field(default_factory=list)
    layer_wall_times: list[float] = field(default_factory=list)
    summarization_s: float = 0.0
    total_s: float = 0.0

    @property
    def execution_s(self) -> float:
        if self.mode == "parallel" and self.layer_wall_times:
            return sum(self.layer_wall_times)
        return sum(s.total_s for s in self.steps)


# ══════════════════════════════════════════════════════════════════════════════
#  Instrumented runner
# ══════════════════════════════════════════════════════════════════════════════


class ProfiledRunner:
    """Wraps PlanExecuteRunner and injects timing at each phase boundary."""

    def __init__(self, model_id: str, server_paths: dict | None = None) -> None:
        from llm.litellm import LiteLLMBackend
        from agent.plan_execute.executor import (
            Executor,
            DEFAULT_SERVER_PATHS,
            _resolve_args_with_llm,
            _call_tool,
            _list_tools,
        )
        from agent.plan_execute.planner import Planner

        self._model_id = model_id
        self._llm = LiteLLMBackend(model_id=model_id)
        self._server_paths = server_paths or DEFAULT_SERVER_PATHS
        self._planner = Planner(self._llm)
        self._executor = Executor(self._llm, self._server_paths)
        self._cache = DiscoveryCache(self._server_paths)

        self._resolve_args_with_llm = _resolve_args_with_llm
        self._call_tool = _call_tool
        self._list_tools = _list_tools

    async def run(
        self,
        question: str,
        parallel: bool = False,
        cache_discovery: bool = False,
    ) -> RunTiming:
        from agent.plan_execute.models import StepResult

        timing = RunTiming(
            question=question,
            mode="parallel" if parallel else "sequential",
            cache_discovery=cache_discovery,
        )
        run_start = time.perf_counter()

        # ── 1. Discovery ──────────────────────────────────────────────
        t0 = time.perf_counter()
        if cache_discovery:
            cached = self._cache.load()
            if cached is not None:
                server_descriptions = cached
            else:
                server_descriptions = await self._executor.get_server_descriptions()
                self._cache.save(server_descriptions)
        else:
            server_descriptions = await self._executor.get_server_descriptions()
        timing.discovery_s = time.perf_counter() - t0

        # ── 2. Planning ───────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self._planner.generate_plan(question, server_descriptions)
        timing.planning_s = time.perf_counter() - t0

        # ── shared: pre-fetch tool schemas ────────────────────────────
        all_steps = plan.steps
        server_names = {step.server for step in all_steps}
        tool_schemas: dict[str, dict[str, str]] = {}
        for name in server_names:
            path = self._server_paths.get(name)
            if path is None:
                continue
            try:
                tools = await self._list_tools(path)
                tool_schemas[name] = {
                    t["name"]: ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    for t in tools
                }
            except Exception:  # noqa: BLE001
                tool_schemas[name] = {}

        context: dict[int, StepResult] = {}

        if parallel:
            # ── 3a. Parallel execution (DAG layer by layer) ───────────
            layers = plan.dependency_layers()
            for layer_idx, layer in enumerate(layers):
                layer_start = time.perf_counter()

                async def _timed_step(step, ctx=context):
                    schema = tool_schemas.get(step.server, {}).get(step.tool, "")
                    return await self._execute_step_timed(
                        step, ctx, question, schema, tool_schemas
                    )

                layer_results: list[tuple[StepTiming, StepResult]] = (
                    await asyncio.gather(*[_timed_step(step) for step in layer])
                )
                layer_wall = time.perf_counter() - layer_start
                timing.layer_wall_times.append(layer_wall)

                for st, result in layer_results:
                    context[result.step_number] = result
                    timing.steps.append(st)
        else:
            # ── 3b. Sequential execution (one step at a time) ─────────
            for step in plan.resolved_order():
                schema = tool_schemas.get(step.server, {}).get(step.tool, "")
                st, result = await self._execute_step_timed(
                    step, context, question, schema, tool_schemas
                )
                context[step.step_number] = result
                timing.steps.append(st)

        # ── 4. Summarization ──────────────────────────────────────────
        from agent.plan_execute.runner import _SUMMARIZE_PROMPT

        results_text = "\n\n".join(
            f"Step {r.step_number} — {r.task} (server: {r.server}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in context.values()
        )
        t0 = time.perf_counter()
        self._llm.generate(
            _SUMMARIZE_PROMPT.format(question=question, results=results_text)
        )
        timing.summarization_s = time.perf_counter() - t0

        timing.total_s = time.perf_counter() - run_start
        return timing

    # ── internal timed step helper ────────────────────────────────────

    async def _execute_step_timed(
        self,
        step,
        context: dict,
        question: str,
        tool_schema: str,
        tool_schemas: dict,
    ):
        """Execute one step and return (StepTiming, StepResult)."""
        from agent.plan_execute.models import StepResult

        step_start = time.perf_counter()
        st = StepTiming(
            step_number=step.step_number,
            server=step.server,
            task=step.task,
            tool=step.tool or "none",
        )

        server_path = self._server_paths.get(step.server)
        if server_path is None or not step.tool or step.tool.lower() in ("none", "null"):
            result = StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=step.expected_output,
                tool=step.tool,
                tool_args=step.tool_args,
            )
            st.total_s = time.perf_counter() - step_start
            return st, result

        try:
            t_llm = time.perf_counter()
            resolved_args = await self._resolve_args_with_llm(
                question, step.task, step.tool, tool_schema, context, self._llm
            )
            st.llm_resolve_s = time.perf_counter() - t_llm

            t_tool = time.perf_counter()
            response = await self._call_tool(server_path, step.tool, resolved_args)
            st.tool_call_s = time.perf_counter() - t_tool

            result = StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=response,
                tool=step.tool,
                tool_args=resolved_args,
            )
        except Exception as exc:  # noqa: BLE001
            result = StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=str(exc),
                tool=step.tool,
                tool_args=step.tool_args,
            )
            st.success = False

        st.total_s = time.perf_counter() - step_start
        return st, result


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting — single-run and multi-run helpers
# ══════════════════════════════════════════════════════════════════════════════


def _bar(value: float, total: float, width: int = 20) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "\u2588" * filled + "\u2591" * (width - filled)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _stats_str(values: list[float]) -> str:
    mn, mx, avg = min(values), max(values), _avg(values)
    return f"avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s"


def print_run(timing: RunTiming, run_index: int | None = None) -> None:
    """Print a detailed single-run breakdown."""
    cache_tag = " +CACHE" if timing.cache_discovery else ""
    mode_label = f"[{timing.mode.upper()}{cache_tag}]"
    label = f"Run {run_index} {mode_label}" if run_index is not None else mode_label
    print(f"\n{'=' * 62}")
    print(f"  {label}: {timing.question[:55]}")
    print(f"{'=' * 62}")

    col = 32
    rows: list[tuple[str, float]] = [
        ("Discovery",      timing.discovery_s),
        ("Planning (LLM)", timing.planning_s),
    ]

    if timing.mode == "parallel" and timing.layer_wall_times:
        for i, lw in enumerate(timing.layer_wall_times):
            rows.append((f"Layer {i + 1} (parallel wall)", lw))
        for st in timing.steps:
            ok = "+" if st.success else "x"
            rows.append((f"  {ok} Step {st.step_number} [{st.server}] {st.tool}", st.total_s))
            if st.llm_resolve_s > 0:
                rows.append(("    -- LLM resolve", st.llm_resolve_s))
            if st.tool_call_s > 0:
                rows.append(("    -- tool call",   st.tool_call_s))
        rows.append(("Execution wall (sum of layers)", timing.execution_s))
    else:
        for st in timing.steps:
            rows.append((f"Step {st.step_number} [{st.server}] {st.tool}", st.total_s))
            if st.llm_resolve_s > 0:
                rows.append(("  -- LLM resolve", st.llm_resolve_s))
            if st.tool_call_s > 0:
                rows.append(("  -- tool call",   st.tool_call_s))

    rows.append(("Summarization (LLM)", timing.summarization_s))

    for lbl, t in rows:
        bar = _bar(t, timing.total_s)
        print(f"  {lbl:<{col}} {t:6.3f}s  {bar}")

    print(f"  {'-' * (col + 30)}")
    print(f"  {'TOTAL':<{col}} {timing.total_s:6.3f}s")

    max_llm = max(
        (timing.planning_s, timing.summarization_s,
         *(st.llm_resolve_s for st in timing.steps)),
        default=0.0,
    )
    if max_llm > 10.0:
        print(f"  ** WatsonX outlier detected: slowest LLM call took {max_llm:.1f}s **")


def print_comparison(seq: RunTiming, par: RunTiming) -> None:
    """Side-by-side diff table after --compare."""
    print(f"\n{'=' * 70}")
    print("  COMPARISON  (sequential vs parallel)")
    print(f"{'=' * 70}")
    col = 26
    print(f"  {'Phase':<{col}}  {'Sequential':>10}  {'Parallel':>10}  {'d (par-seq)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 66}")
    phases = [
        ("Discovery",          seq.discovery_s,     par.discovery_s),
        ("Planning (LLM)",     seq.planning_s,      par.planning_s),
        ("Execution (total)",  seq.execution_s,     par.execution_s),
        ("Summarization (LLM)",seq.summarization_s, par.summarization_s),
        ("TOTAL",              seq.total_s,         par.total_s),
    ]
    for name, s, p in phases:
        delta = p - s
        speedup = s / p if p > 0 else float("inf")
        delta_str = f"{delta:+.3f}s"
        speedup_str = f"{speedup:.2f}x"
        print(f"  {name:<{col}}  {s:>10.3f}s  {p:>10.3f}s  {delta_str:>12}  {speedup_str:>8}")
    saved = seq.total_s - par.total_s
    pct = saved / seq.total_s * 100 if seq.total_s > 0 else 0
    print(f"  {'-' * 66}")
    print(f"  Time saved: {saved:.3f}s  ({pct:.1f}% faster with parallel)")


def print_summary(timings: list[RunTiming]) -> None:
    if len(timings) < 2:
        return

    print(f"\n{'=' * 62}")
    print(f"  Summary across {len(timings)} runs")
    print(f"{'=' * 62}")

    col = 28
    print(f"  {'Phase':<{col}}  avg      min      max")
    print(f"  {'-' * 56}")

    phases = [
        ("Discovery",          [t.discovery_s for t in timings]),
        ("Planning (LLM)",     [t.planning_s for t in timings]),
        ("Execution (total)",  [t.execution_s for t in timings]),
        ("Summarization (LLM)",[t.summarization_s for t in timings]),
        ("TOTAL",              [t.total_s for t in timings]),
    ]
    for name, values in phases:
        print(f"  {name:<{col}}  {_stats_str(values)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Optimization benchmark — reporting helpers
# ══════════════════════════════════════════════════════════════════════════════


def _print_opt_comparison(
    title: str,
    left_label: str,
    right_label: str,
    left: list[RunTiming],
    right: list[RunTiming],
) -> None:
    """Print a median-based comparison table between two timing groups.

    Median is used instead of mean because WatsonX LLM latency has extreme
    variance (same call can take 0.5s or 40s).  Median is resistant to
    these outliers and gives the "typical" run performance.
    """

    def med_phase(timings: list[RunTiming], attr: str) -> float:
        return _median([getattr(t, attr) for t in timings])

    def med_exec(timings: list[RunTiming]) -> float:
        return _median([t.execution_s for t in timings])

    n = min(len(left), len(right))
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"  (median of {n} runs per mode — robust to WatsonX outliers)")
    print(f"{'=' * 78}")

    col = 22
    print(
        f"  {'Phase':<{col}} "
        f"{left_label:>12} "
        f"{right_label:>12} "
        f"{'Saving':>10} "
        f"{'Speedup':>8}"
    )
    print(f"  {'-' * 72}")

    phases = [
        ("Discovery",           med_phase(left, "discovery_s"),     med_phase(right, "discovery_s")),
        ("Planning (LLM)",      med_phase(left, "planning_s"),      med_phase(right, "planning_s")),
        ("Execution (total)",   med_exec(left),                     med_exec(right)),
        ("Summarization (LLM)", med_phase(left, "summarization_s"), med_phase(right, "summarization_s")),
        ("TOTAL",               med_phase(left, "total_s"),         med_phase(right, "total_s")),
    ]
    for name, lv, rv in phases:
        saving = lv - rv
        speedup = lv / rv if rv > 0 else float("inf")
        saving_str = f"{saving:+.3f}s" if abs(saving) > 0.005 else "~0"
        speedup_str = f"{speedup:.2f}x" if abs(saving) > 0.005 else "-"
        print(f"  {name:<{col}} {lv:>11.3f}s {rv:>11.3f}s {saving_str:>10} {speedup_str:>8}")

    total_left = med_phase(left, "total_s")
    total_right = med_phase(right, "total_s")
    saved = total_left - total_right
    pct = saved / total_left * 100 if total_left > 0 else 0
    print(f"  {'-' * 72}")
    print(f"  Latency cut: {saved:.3f}s ({pct:.1f}% faster)")


def _print_mode_summary(label: str, timings: list[RunTiming]) -> None:
    """Print summary for a single mode's runs with median as primary metric."""
    n = len(timings)
    print(f"\n  {label} (over {n} runs):")
    col = 26
    print(f"    {'Phase':<{col}}  {'Median':>8}  {'Avg':>8}  {'Min':>8}  {'Max':>8}")
    print(f"    {'-' * 66}")
    phases = [
        ("Discovery",          [t.discovery_s for t in timings]),
        ("Planning (LLM)",     [t.planning_s for t in timings]),
        ("Execution (total)",  [t.execution_s for t in timings]),
        ("Summarization (LLM)",[t.summarization_s for t in timings]),
        ("TOTAL",              [t.total_s for t in timings]),
    ]
    for name, vals in phases:
        med, avg, mn, mx = _median(vals), _avg(vals), min(vals), max(vals)
        print(f"    {name:<{col}}  {med:>7.3f}s  {avg:>7.3f}s  {mn:>7.3f}s  {mx:>7.3f}s")


# ══════════════════════════════════════════════════════════════════════════════
#  Optimization benchmark — main entry point
# ══════════════════════════════════════════════════════════════════════════════


_MAX_RETRIES = 3
_RETRY_BACKOFF = 5  # seconds between retries


async def _run_with_retry(
    runner: ProfiledRunner,
    question: str,
    parallel: bool,
    cache_discovery: bool,
    run_label: str,
) -> RunTiming | None:
    """Execute a single profiled run with retry on transient WatsonX errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await runner.run(
                question, parallel=parallel, cache_discovery=cache_discovery
            )
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            is_transient = any(
                k in err_str
                for k in ("InternalServerError", "500", "connection refused", "timed out")
            )
            if is_transient and attempt < _MAX_RETRIES:
                print(
                    f"    WatsonX transient error (attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {_RETRY_BACKOFF}s...",
                    flush=True,
                )
                await asyncio.sleep(_RETRY_BACKOFF)
            else:
                print(f"    Run failed: {err_str[:120]}", flush=True)
                return None


async def run_optimization(
    question: str,
    model_id: str = "watsonx/meta-llama/llama-3-3-70b-instruct",
    runs: int = 3,
) -> None:
    """Run the MCP workflow optimization benchmark.

    Executes the query in two modes, each repeated ``runs`` times,
    and reports the end-to-end latency improvement:

      Baseline  — sequential execution, no discovery cache
      Optimized — parallel execution  + discovery cache

    Each run retries up to 3 times on transient WatsonX errors.
    Failed runs are skipped; the report uses only successful runs.
    """
    runner = ProfiledRunner(model_id=model_id)

    print(f"\n{'=' * 78}")
    print("  MCP WORKFLOW OPTIMIZATION BENCHMARK")
    print(f"{'=' * 78}")
    print(f"  Query:  {question}")
    print(f"  Model:  {model_id}")
    print(f"  Runs:   {runs} per mode (2 modes = {runs * 2} total executions)")
    print(f"{'=' * 78}")

    # ── Baseline (sequential, no cache) ───────────────────────────────
    print(f"\n{'-' * 78}")
    print("  BASELINE (sequential, no discovery cache)")
    print(f"{'-' * 78}")
    runner._cache.clear()
    baseline: list[RunTiming] = []
    for i in range(1, runs + 1):
        print(f"\n  >> Baseline run {i}/{runs}...", flush=True)
        t = await _run_with_retry(
            runner, question, parallel=False, cache_discovery=False,
            run_label=f"Baseline {i}/{runs}",
        )
        if t is not None:
            baseline.append(t)
            print_run(t, run_index=i)
        else:
            print(f"  >> Baseline run {i}/{runs} SKIPPED (WatsonX error).")

    if baseline:
        _print_mode_summary("Baseline", baseline)
    else:
        print("\n  All baseline runs failed. Cannot produce comparison.")
        return

    # ── Optimized (parallel + cached discovery) ───────────────────────
    print(f"\n{'-' * 78}")
    print("  OPTIMIZED (parallel + cached discovery)")
    print(f"{'-' * 78}")
    print("  Priming discovery cache...", flush=True)
    desc = await runner._executor.get_server_descriptions()
    runner._cache.save(desc)
    print("  Cache primed.")

    optimized: list[RunTiming] = []
    for i in range(1, runs + 1):
        print(f"\n  >> Optimized run {i}/{runs}...", flush=True)
        t = await _run_with_retry(
            runner, question, parallel=True, cache_discovery=True,
            run_label=f"Optimized {i}/{runs}",
        )
        if t is not None:
            optimized.append(t)
            print_run(t, run_index=i)
        else:
            print(f"  >> Optimized run {i}/{runs} SKIPPED (WatsonX error).")

    if optimized:
        _print_mode_summary("Optimized", optimized)
    else:
        print("\n  All optimized runs failed. Cannot produce comparison.")
        return

    # ══════════════════════════════════════════════════════════════════
    #  End-to-end comparison
    # ══════════════════════════════════════════════════════════════════

    _print_opt_comparison(
        "END-TO-END: Baseline vs Optimized (Cache + Parallel)",
        "Baseline",
        "Optimized",
        baseline,
        optimized,
    )

    total_baseline = _median([t.total_s for t in baseline])
    total_optimized = _median([t.total_s for t in optimized])
    total_saving = total_baseline - total_optimized

    if total_optimized > 0 and total_baseline > 0:
        overall_pct = total_saving / total_baseline * 100
        print(f"\n  Overall (median): {total_saving:+.3f}s saved "
              f"({total_baseline / total_optimized:.2f}x speedup, "
              f"{overall_pct:.1f}% faster)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler",
        description="Integrated MCP workflow optimization profiler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Optimization benchmark (5 runs per mode, 10 total — uses median)
  uv run python timer.py --optimize "What assets are at site MAIN?"
  uv run python timer.py --optimize --runs 7 "query"

  # Individual profiling
  uv run python timer.py "query"                          # baseline
  uv run python timer.py --cache-discovery "query"        # with cache
  uv run python timer.py --parallel "query"               # parallel
  uv run python timer.py --cache-discovery --parallel "q" # both
  uv run python timer.py --compare "query"                # seq vs par
  uv run python timer.py --runs 3 "query"                 # multi-run avg

  # Cache management
  uv run python timer.py --clear-cache                    # wipe cache
""",
    )
    parser.add_argument("question", nargs="?", default="", help="The question to profile.")
    parser.add_argument(
        "--model-id",
        default="watsonx/meta-llama/llama-3-3-70b-instruct",
        metavar="MODEL_ID",
        help="LiteLLM model string (default: watsonx/meta-llama/llama-3-3-70b-instruct).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of runs per mode (default: 1; 3+ recommended for stable averages).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--parallel",
        action="store_true",
        help="Run with parallel (DAG) executor instead of sequential.",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Run BOTH sequential and parallel back-to-back and print a comparison.",
    )
    mode.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Run the optimization benchmark: baseline vs optimized "
            "(cache + parallel), each repeated --runs times (default 5). "
            "Uses median for comparison (robust to WatsonX outliers). "
            "Retries on transient WatsonX errors."
        ),
    )
    parser.add_argument(
        "--cache-discovery",
        action="store_true",
        help="Enable discovery phase caching (disk-backed, 24h TTL).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the discovery cache file and exit.",
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    # ── Clear cache and exit ──────────────────────────────────────────
    if args.clear_cache:
        from agent.plan_execute.executor import DEFAULT_SERVER_PATHS
        cache = DiscoveryCache(DEFAULT_SERVER_PATHS)
        cache.clear()
        return

    if not args.question:
        print("error: question is required (unless using --clear-cache)", flush=True)
        return

    # ── Full optimization benchmark ───────────────────────────────────
    if args.optimize:
        runs = args.runs if args.runs > 1 else 5
        await run_optimization(args.question, args.model_id, runs)
        return

    # ── Individual profiling modes (original behaviour) ───────────────
    runner = ProfiledRunner(model_id=args.model_id)

    if args.compare:
        print("\n>> Running SEQUENTIAL...", flush=True)
        seq = await runner.run(
            args.question, parallel=False, cache_discovery=args.cache_discovery
        )
        print_run(seq)

        print("\n>> Running PARALLEL...", flush=True)
        par = await runner.run(
            args.question, parallel=True, cache_discovery=args.cache_discovery
        )
        print_run(par)

        print_comparison(seq, par)
        return

    timings: list[RunTiming] = []
    for i in range(1, args.runs + 1):
        if args.runs > 1:
            mode_name = "parallel" if args.parallel else "sequential"
            cache_name = " +cache" if args.cache_discovery else ""
            print(f"\nRun {i}/{args.runs} ({mode_name}{cache_name})...", flush=True)
        t = await runner.run(
            args.question,
            parallel=args.parallel,
            cache_discovery=args.cache_discovery,
        )
        timings.append(t)
        print_run(t, run_index=i if args.runs > 1 else None)

    print_summary(timings)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
