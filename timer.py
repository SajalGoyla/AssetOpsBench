"""Latency profiler for the plan-execute workflow.

Measures wall-clock time at each phase:
  1. Discovery  — spawning MCP servers and listing tools
  2. Planning   — LLM call to decompose the question into steps
  3. Execution  — sequential (one step at a time) OR parallel (DAG layers)
  4. Summary    — LLM call to synthesise the final answer

Usage:
    uv run python timer.py "What assets are at site MAIN?"
    uv run python timer.py --parallel "What assets are at site MAIN?"
    uv run python timer.py --compare "What assets are at site MAIN?"
    uv run python timer.py --runs 3 "What assets are at site MAIN?"
    uv run python timer.py --model-id watsonx/ibm/granite-3-3-8b-instruct "..."
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

# ── data classes ──────────────────────────────────────────────────────────────

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
    discovery_s: float = 0.0
    planning_s: float = 0.0
    steps: list[StepTiming] = field(default_factory=list)
    # Parallel mode: wall time per DAG layer (avoids N-fold over-count).
    layer_wall_times: list[float] = field(default_factory=list)
    summarization_s: float = 0.0
    total_s: float = 0.0

    @property
    def execution_s(self) -> float:
        # Sequential: sum of individual step times.
        # Parallel: sum of *layer* wall times (steps in the same layer run
        # concurrently, so summing their individual times would over-count).
        if self.mode == "parallel" and self.layer_wall_times:
            return sum(self.layer_wall_times)
        return sum(s.total_s for s in self.steps)


# ── instrumented runner ───────────────────────────────────────────────────────

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

        # stash references to internal helpers for timed calls
        self._resolve_args_with_llm = _resolve_args_with_llm
        self._call_tool = _call_tool
        self._list_tools = _list_tools

    async def run(self, question: str, parallel: bool = False) -> RunTiming:
        from agent.plan_execute.models import StepResult

        timing = RunTiming(question=question, mode="parallel" if parallel else "sequential")
        run_start = time.perf_counter()

        # ── 1. Discovery ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        server_descriptions = await self._executor.get_server_descriptions()
        timing.discovery_s = time.perf_counter() - t0

        # ── 2. Planning ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self._planner.generate_plan(question, server_descriptions)
        timing.planning_s = time.perf_counter() - t0

        # ── shared: pre-fetch tool schemas ────────────────────────────────────
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
            # ── 3a. Parallel execution (DAG layer by layer) ───────────────────
            layers = plan.dependency_layers()
            for layer_idx, layer in enumerate(layers):
                layer_start = time.perf_counter()

                async def _timed_step(step, ctx=context):
                    schema = tool_schemas.get(step.server, {}).get(step.tool, "")
                    return await self._execute_step_timed(
                        step, ctx, question, schema, tool_schemas
                    )

                layer_results: list[tuple[StepTiming, StepResult]] = await asyncio.gather(
                    *[_timed_step(step) for step in layer]
                )
                layer_wall = time.perf_counter() - layer_start
                # Store the layer wall time ONCE — not per-step.
                # execution_s will sum these, giving true parallel wall time.
                timing.layer_wall_times.append(layer_wall)

                for st, result in layer_results:
                    # Keep individual coroutine time in st.total_s for the
                    # per-step breakdown; layer wall time is tracked separately.
                    context[result.step_number] = result
                    timing.steps.append(st)
        else:
            # ── 3b. Sequential execution (one step at a time) ─────────────────
            for step in plan.resolved_order():
                schema = tool_schemas.get(step.server, {}).get(step.tool, "")
                st, result = await self._execute_step_timed(
                    step, context, question, schema, tool_schemas
                )
                context[step.step_number] = result
                timing.steps.append(st)

        # ── 4. Summarization ──────────────────────────────────────────────────
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

    # ── internal timed step helper ────────────────────────────────────────────

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


# ── reporting ─────────────────────────────────────────────────────────────────

def _bar(value: float, total: float, width: int = 20) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_run(timing: RunTiming, run_index: int | None = None) -> None:
    mode_label = f"[{timing.mode.upper()}]"
    label = f"Run {run_index} {mode_label}" if run_index is not None else mode_label
    print(f"\n{'═' * 62}")
    print(f"  {label}: {timing.question[:55]}")
    print(f"{'═' * 62}")

    col = 32
    rows: list[tuple[str, float]] = [
        ("Discovery",      timing.discovery_s),
        ("Planning (LLM)", timing.planning_s),
    ]

    if timing.mode == "parallel" and timing.layer_wall_times:
        # Show per-layer wall times (the real parallel latency unit).
        for i, lw in enumerate(timing.layer_wall_times):
            rows.append((f"Layer {i + 1} (parallel wall)", lw))
        # Also show individual step breakdown underneath.
        for st in timing.steps:
            ok = "✓" if st.success else "✗"
            rows.append((f"  {ok} Step {st.step_number} [{st.server}] {st.tool}", st.total_s))
            if st.llm_resolve_s > 0:
                rows.append(("    └─ LLM resolve", st.llm_resolve_s))
            if st.tool_call_s > 0:
                rows.append(("    └─ tool call",   st.tool_call_s))
        rows.append((f"Execution wall (sum of layers)", timing.execution_s))
    else:
        for st in timing.steps:
            rows.append((f"Step {st.step_number} [{st.server}] {st.tool}", st.total_s))
            if st.llm_resolve_s > 0:
                rows.append(("  └─ LLM resolve", st.llm_resolve_s))
            if st.tool_call_s > 0:
                rows.append(("  └─ tool call",   st.tool_call_s))

    rows.append(("Summarization (LLM)", timing.summarization_s))

    for lbl, t in rows:
        bar = _bar(t, timing.total_s)
        print(f"  {lbl:<{col}} {t:6.3f}s  {bar}")

    print(f"  {'─' * (col + 30)}")
    print(f"  {'TOTAL':<{col}} {timing.total_s:6.3f}s")


def print_comparison(seq: RunTiming, par: RunTiming) -> None:
    """Side-by-side diff table after --compare."""
    print(f"\n{'═' * 70}")
    print(f"  COMPARISON  (sequential vs parallel)")
    print(f"{'═' * 70}")
    col = 26
    print(f"  {'Phase':<{col}}  {'Sequential':>10}  {'Parallel':>10}  {'Δ (par-seq)':>12}  {'Speedup':>8}")
    print(f"  {'─' * 66}")
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
    print(f"  {'─' * 66}")
    print(f"  Time saved: {saved:.3f}s  ({pct:.1f}% faster with parallel)")


def print_summary(timings: list[RunTiming]) -> None:
    if len(timings) < 2:
        return

    print(f"\n{'═' * 62}")
    print(f"  Summary across {len(timings)} runs")
    print(f"{'═' * 62}")

    def stats(values: list[float]) -> str:
        mn, mx, avg = min(values), max(values), sum(values) / len(values)
        return f"avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s"

    col = 28
    print(f"  {'Phase':<{col}}  avg      min      max")
    print(f"  {'─' * 56}")

    phases = [
        ("Discovery",          [t.discovery_s for t in timings]),
        ("Planning (LLM)",     [t.planning_s for t in timings]),
        ("Execution (total)",  [t.execution_s for t in timings]),
        ("Summarization (LLM)",[t.summarization_s for t in timings]),
        ("TOTAL",              [t.total_s for t in timings]),
    ]
    for name, values in phases:
        print(f"  {name:<{col}}  {stats(values)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler",
        description="Phase-level latency profiler for the plan-execute workflow.",
    )
    parser.add_argument("question", help="The question to profile.")
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
        help="Number of times to run the query (default: 1). Use 3+ for stable averages.",
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
        help="Run BOTH sequential and parallel back-to-back and print a comparison table.",
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    runner = ProfiledRunner(model_id=args.model_id)

    if args.compare:
        # Run sequential first, then parallel, print both + comparison.
        print("\n▶ Running SEQUENTIAL...", flush=True)
        seq = await runner.run(args.question, parallel=False)
        print_run(seq)

        print("\n▶ Running PARALLEL...", flush=True)
        par = await runner.run(args.question, parallel=True)
        print_run(par)

        print_comparison(seq, par)
        return

    timings: list[RunTiming] = []
    for i in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\nRun {i}/{args.runs} ({('parallel' if args.parallel else 'sequential')})...", flush=True)
        t = await runner.run(args.question, parallel=args.parallel)
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
