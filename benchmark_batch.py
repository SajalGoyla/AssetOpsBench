"""Batch stress-test benchmarker for MCP workflow optimizations.

Reads queries from a text file, runs each in Baseline (sequential, no cache)
and Optimized (parallel + discovery cache) modes with N repeats, computes
averaged per-phase metrics, and exports CSV / JSON results.

Usage:
    uv run python benchmark_batch.py --queries queries.txt
    uv run python benchmark_batch.py --queries queries.txt --runs 3 --output-dir results/
    uv run python benchmark_batch.py --queries queries.txt --resume
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Reuse core classes from timer.py (same repo root)
# ---------------------------------------------------------------------------
# We import lazily inside functions so the module-level stays light.

_REPO_ROOT = Path(__file__).resolve().parent
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5


# ═══════════════════════════════════════════════════════════════════════════
#  Query file reader
# ═══════════════════════════════════════════════════════════════════════════

def read_queries(path: str | Path) -> list[str]:
    """Read queries from a text file (one per line, # comments, blank ignored)."""
    queries: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                queries.append(stripped)
    if not queries:
        print(f"ERROR: no queries found in {path}", file=sys.stderr)
        sys.exit(1)
    return queries


# ═══════════════════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"avg": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    import statistics
    return {
        "avg": _avg(vals),
        "median": _median(vals),
        "min": min(vals),
        "max": max(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


@dataclass
class QueryResult:
    """Aggregated result for one query across N runs × 2 modes."""
    query_id: int
    query_text: str
    # Per-phase medians — baseline
    bl_discovery: float = 0.0
    bl_planning: float = 0.0
    bl_prefetch: float = 0.0
    bl_execution: float = 0.0
    bl_summarization: float = 0.0
    bl_total: float = 0.0
    bl_success_rate: float = 0.0
    bl_num_steps: float = 0.0
    # Per-phase medians — optimized
    opt_discovery: float = 0.0
    opt_planning: float = 0.0
    opt_prefetch: float = 0.0
    opt_execution: float = 0.0
    opt_summarization: float = 0.0
    opt_total: float = 0.0
    opt_success_rate: float = 0.0
    opt_num_steps: float = 0.0
    # Speedups
    discovery_speedup: float = 0.0
    execution_speedup: float = 0.0
    total_speedup: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Single-run with retry
# ═══════════════════════════════════════════════════════════════════════════
_RUN_TIMEOUT = 180  # seconds — max wall-clock time for a single profiled run


async def _run_with_retry(runner, question, parallel, cache_discovery, label):
    """Execute one profiled run with retry on transient errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await asyncio.wait_for(
                runner.run(
                    question, parallel=parallel, cache_discovery=cache_discovery
                ),
                timeout=_RUN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            if attempt < _MAX_RETRIES:
                print(f"      TIMEOUT ({label} attempt {attempt}, >{_RUN_TIMEOUT}s), retrying in {_RETRY_BACKOFF}s...")
                await asyncio.sleep(_RETRY_BACKOFF)
            else:
                print(f"      TIMEOUT ({label}): exceeded {_RUN_TIMEOUT}s on all attempts")
                return None
        except Exception as exc:
            err = str(exc)
            transient = any(k in err for k in ("500", "InternalServerError", "timed out", "connection refused", "Timeout"))
            if transient and attempt < _MAX_RETRIES:
                print(f"      Transient error ({label} attempt {attempt}), retrying in {_RETRY_BACKOFF}s...")
                await asyncio.sleep(_RETRY_BACKOFF)
            else:
                print(f"      FAILED ({label}): {err[:120]}")
                return None


# ═══════════════════════════════════════════════════════════════════════════
#  Core benchmark loop
# ═══════════════════════════════════════════════════════════════════════════

def _timing_to_dict(t) -> dict[str, Any]:
    """Serialize a RunTiming to a JSON-safe dict."""
    return {
        "question": t.question,
        "mode": t.mode,
        "cache_discovery": t.cache_discovery,
        "discovery_s": t.discovery_s,
        "planning_s": t.planning_s,
        "prefetch_s": t.prefetch_s,
        "execution_s": t.execution_s,
        "summarization_s": t.summarization_s,
        "total_s": t.total_s,
        "num_steps": len(t.steps),
        "plan": [
            {
                "step_number": ps.step_number,
                "task": ps.task,
                "server": ps.server,
                "tool": ps.tool,
                "dependencies": ps.dependencies,
            }
            for ps in t.plan_steps
        ],
        "plan_layers": t.plan_layers,
        "steps": [
            {
                "step_number": s.step_number,
                "server": s.server,
                "task": s.task,
                "tool": s.tool,
                "tool_args": s.tool_args,
                "response": s.response,
                "error": s.error,
                "llm_resolve_s": s.llm_resolve_s,
                "tool_call_s": s.tool_call_s,
                "total_s": s.total_s,
                "success": s.success,
            }
            for s in t.steps
        ],
        "answer": t.answer,
    }


def _compute_query_result(
    qid: int, query: str, baseline_runs: list, optimized_runs: list
) -> QueryResult:
    """Compute metrics for one query using median (robust to WatsonX outliers)."""
    qr = QueryResult(query_id=qid, query_text=query)

    if baseline_runs:
        qr.bl_discovery = _median([t.discovery_s for t in baseline_runs])
        qr.bl_planning = _median([t.planning_s for t in baseline_runs])
        qr.bl_prefetch = _median([t.prefetch_s for t in baseline_runs])
        qr.bl_execution = _median([t.execution_s for t in baseline_runs])
        qr.bl_summarization = _median([t.summarization_s for t in baseline_runs])
        qr.bl_total = _median([t.total_s for t in baseline_runs])
        qr.bl_num_steps = _median([len(t.steps) for t in baseline_runs])
        qr.bl_success_rate = len(baseline_runs)  # successful runs count

    if optimized_runs:
        qr.opt_discovery = _median([t.discovery_s for t in optimized_runs])
        qr.opt_planning = _median([t.planning_s for t in optimized_runs])
        qr.opt_prefetch = _median([t.prefetch_s for t in optimized_runs])
        qr.opt_execution = _median([t.execution_s for t in optimized_runs])
        qr.opt_summarization = _median([t.summarization_s for t in optimized_runs])
        qr.opt_total = _median([t.total_s for t in optimized_runs])
        qr.opt_num_steps = _median([len(t.steps) for t in optimized_runs])
        qr.opt_success_rate = len(optimized_runs)

    # Speedups (baseline / optimized)
    qr.discovery_speedup = qr.bl_discovery / qr.opt_discovery if qr.opt_discovery > 0 else 0
    qr.execution_speedup = qr.bl_execution / qr.opt_execution if qr.opt_execution > 0 else 0
    qr.total_speedup = qr.bl_total / qr.opt_total if qr.opt_total > 0 else 0
    return qr


def _load_completed_ids(output_dir: Path) -> set[int]:
    """Load query IDs already completed from raw_runs.json for resume."""
    raw_path = output_dir / "raw_runs.json"
    if not raw_path.exists():
        return set()
    try:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        return {entry["query_id"] for entry in data}
    except Exception:
        return set()


def _load_raw_runs(output_dir: Path) -> list[dict]:
    raw_path = output_dir / "raw_runs.json"
    if not raw_path.exists():
        return []
    try:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return []


async def run_benchmark(
    queries: list[str],
    runs: int,
    model_id: str,
    output_dir: Path,
    cooldown: float,
    resume: bool,
) -> None:
    """Main benchmark entry point."""
    # Lazy imports so CLI --help stays fast
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from timer import ProfiledRunner, DiscoveryCache

    output_dir.mkdir(parents=True, exist_ok=True)
    runner = ProfiledRunner(model_id=model_id)

    total_queries = len(queries)
    total_runs = total_queries * runs * 2

    print(f"\n{'=' * 78}")
    print("  ASSETOPSBENCH BATCH STRESS-TEST BENCHMARK")
    print(f"{'=' * 78}")
    print(f"  Queries:      {total_queries}")
    print(f"  Runs/query:   {runs} per mode (baseline + optimized)")
    print(f"  Total runs:   {total_runs}")
    print(f"  Model:        {model_id}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Resume:       {resume}")
    print(f"{'=' * 78}\n")

    # --- Prime discovery cache once for all optimized runs ---
    print("  Priming discovery cache...", flush=True)
    desc = await runner._executor.get_server_descriptions()
    runner._cache.save(desc)
    print("  Cache primed.\n")

    # --- Resume support ---
    completed_ids: set[int] = set()
    all_raw: list[dict] = []
    if resume:
        completed_ids = _load_completed_ids(output_dir)
        all_raw = _load_raw_runs(output_dir)
        if completed_ids:
            print(f"  Resuming — {len(completed_ids)} queries already done, skipping.\n")

    query_results: list[QueryResult] = []
    completed_count = len(completed_ids)

    for qid, query in enumerate(queries, start=1):
        if qid in completed_ids:
            continue

        completed_count += 1
        print(f"\n{'─' * 78}")
        print(f"  Query {completed_count}/{total_queries} (ID={qid})")
        print(f"  \"{query[:70]}{'...' if len(query) > 70 else ''}\"")
        print(f"{'─' * 78}")

        # ── BASELINE runs (sequential, no cache) ──
        baseline_runs = []
        for i in range(1, runs + 1):
            runner._cache.clear()  # wipe cache before every baseline run
            print(f"    Baseline run {i}/{runs}...", end=" ", flush=True)
            t = await _run_with_retry(
                runner, query, parallel=False, cache_discovery=False,
                label=f"BL-{i}",
            )
            if t is not None:
                baseline_runs.append(t)
                print(f"OK ({t.total_s:.2f}s)")
            else:
                print("SKIPPED")

        # ── OPTIMIZED runs (parallel + cache) ──
        # Re-prime cache (baseline clears it)
        runner._cache.save(desc)
        optimized_runs = []
        for i in range(1, runs + 1):
            print(f"    Optimized run {i}/{runs}...", end=" ", flush=True)
            t = await _run_with_retry(
                runner, query, parallel=True, cache_discovery=True,
                label=f"OPT-{i}",
            )
            if t is not None:
                optimized_runs.append(t)
                print(f"OK ({t.total_s:.2f}s)")
            else:
                print("SKIPPED")

        # ── Compute per-query result ──
        qr = _compute_query_result(qid, query, baseline_runs, optimized_runs)
        query_results.append(qr)

        # Print quick per-query summary
        print(f"\n    {'Phase':<20} {'Baseline':>10} {'Optimized':>10} {'Speedup':>8}")
        print(f"    {'─' * 52}")
        for label, bl, opt in [
            ("Discovery",      qr.bl_discovery,      qr.opt_discovery),
            ("Planning",       qr.bl_planning,       qr.opt_planning),
            ("Pre-fetch",      qr.bl_prefetch,       qr.opt_prefetch),
            ("Execution",      qr.bl_execution,      qr.opt_execution),
            ("Summarization",  qr.bl_summarization,  qr.opt_summarization),
            ("Completion",     qr.bl_total,           qr.opt_total),
        ]:
            sp = bl / opt if opt > 0 else 0
            print(f"    {label:<20} {bl:>9.3f}s {opt:>9.3f}s {sp:>7.2f}x")

        # ── Incremental save (raw runs) ──
        raw_entry = {
            "query_id": qid,
            "query_text": query,
            "baseline": [_timing_to_dict(t) for t in baseline_runs],
            "optimized": [_timing_to_dict(t) for t in optimized_runs],
        }
        all_raw.append(raw_entry)
        (output_dir / "raw_runs.json").write_text(
            json.dumps(all_raw, indent=2), encoding="utf-8"
        )

        # Cooldown
        if cooldown > 0 and qid < total_queries:
            print(f"\n    Cooldown {cooldown}s...", flush=True)
            await asyncio.sleep(cooldown)

    # ═══════════════════════════════════════════════════════════════════
    #  Final outputs
    # ═══════════════════════════════════════════════════════════════════

    # --- Also include resumed results for aggregation ---
    if resume and completed_ids:
        for raw_entry in _load_raw_runs(output_dir):
            rid = raw_entry["query_id"]
            if rid in completed_ids:
                # Reconstruct QueryResult from raw data
                bl = raw_entry.get("baseline", [])
                opt = raw_entry.get("optimized", [])
                # We stored timing dicts; build a lightweight stand-in
                qr = QueryResult(query_id=rid, query_text=raw_entry["query_text"])
                if bl:
                    qr.bl_discovery = _avg([r["discovery_s"] for r in bl])
                    qr.bl_planning = _avg([r["planning_s"] for r in bl])
                    qr.bl_prefetch = _avg([r.get("prefetch_s", 0) for r in bl])
                    qr.bl_execution = _avg([r["execution_s"] for r in bl])
                    qr.bl_summarization = _avg([r["summarization_s"] for r in bl])
                    qr.bl_total = _avg([r["total_s"] for r in bl])
                    qr.bl_num_steps = _avg([r["num_steps"] for r in bl])
                    qr.bl_success_rate = len(bl)
                if opt:
                    qr.opt_discovery = _avg([r["discovery_s"] for r in opt])
                    qr.opt_planning = _avg([r["planning_s"] for r in opt])
                    qr.opt_prefetch = _avg([r.get("prefetch_s", 0) for r in opt])
                    qr.opt_execution = _avg([r["execution_s"] for r in opt])
                    qr.opt_summarization = _avg([r["summarization_s"] for r in opt])
                    qr.opt_total = _avg([r["total_s"] for r in opt])
                    qr.opt_num_steps = _avg([r["num_steps"] for r in opt])
                    qr.opt_success_rate = len(opt)
                qr.discovery_speedup = qr.bl_discovery / qr.opt_discovery if qr.opt_discovery > 0 else 0
                qr.execution_speedup = qr.bl_execution / qr.opt_execution if qr.opt_execution > 0 else 0
                qr.total_speedup = qr.bl_total / qr.opt_total if qr.opt_total > 0 else 0
                query_results.append(qr)

    # Sort by query_id
    query_results.sort(key=lambda q: q.query_id)

    _write_csv(query_results, output_dir)
    _write_aggregate_json(query_results, runs, model_id, output_dir)
    _write_summary_report(query_results, runs, model_id, output_dir)

    print(f"\n{'=' * 78}")
    print(f"  BENCHMARK COMPLETE — results in {output_dir}/")
    print(f"{'=' * 78}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Output writers
# ═══════════════════════════════════════════════════════════════════════════

_CSV_COLUMNS = [
    "query_id", "query_text",
    "bl_discovery", "bl_planning", "bl_prefetch", "bl_execution", "bl_summarization", "bl_total",
    "bl_success_rate", "bl_num_steps",
    "opt_discovery", "opt_planning", "opt_prefetch", "opt_execution", "opt_summarization", "opt_total",
    "opt_success_rate", "opt_num_steps",
    "discovery_speedup", "execution_speedup", "total_speedup",
]


def _write_csv(results: list[QueryResult], output_dir: Path) -> None:
    path = output_dir / "per_query_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for qr in results:
            row = {k: getattr(qr, k) for k in _CSV_COLUMNS}
            # Round floats
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = round(v, 4)
            w.writerow(row)
    print(f"  CSV  → {path}")


def _write_aggregate_json(
    results: list[QueryResult], runs: int, model_id: str, output_dir: Path
) -> None:
    agg: dict[str, Any] = {
        "num_queries": len(results),
        "runs_per_query": runs,
        "model_id": model_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    phases = ["discovery", "planning", "prefetch", "execution", "summarization", "total"]
    for mode_prefix, mode_label in [("bl_", "baseline"), ("opt_", "optimized")]:
        mode_stats: dict[str, Any] = {}
        for phase in phases:
            vals = [getattr(qr, f"{mode_prefix}{phase}") for qr in results]
            mode_stats[phase] = _stats(vals)
        agg[mode_label] = mode_stats

    speedup_stats: dict[str, Any] = {}
    for phase in ["discovery", "execution", "total"]:
        vals = [getattr(qr, f"{phase}_speedup") for qr in results]
        speedup_stats[phase] = _stats(vals)
    agg["speedups"] = speedup_stats

    path = output_dir / "aggregate_summary.json"
    path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"  JSON → {path}")


def _write_summary_report(
    results: list[QueryResult], runs: int, model_id: str, output_dir: Path
) -> None:
    lines: list[str] = []
    w = 78

    lines.append("=" * w)
    lines.append("  ASSETOPSBENCH BATCH BENCHMARK — SUMMARY REPORT")
    lines.append("=" * w)
    lines.append(f"  Queries:       {len(results)}")
    lines.append(f"  Runs/query:    {runs} per mode")
    lines.append(f"  Model:         {model_id}")
    lines.append(f"  Generated:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * w)

    # Aggregate table
    lines.append("")
    lines.append(f"  {'Phase':<20} {'Baseline (med)':>14} {'Optimized (med)':>16} {'Speedup':>8}")
    lines.append(f"  {'─' * 62}")

    phases = ["discovery", "planning", "prefetch", "execution", "summarization", "total"]
    label_map = {"prefetch": "Pre-fetch", "total": "Completion"}
    for phase in phases:
        bl_vals = [getattr(qr, f"bl_{phase}") for qr in results]
        opt_vals = [getattr(qr, f"opt_{phase}") for qr in results]
        bl_avg = _avg(bl_vals)
        opt_avg = _avg(opt_vals)
        sp = bl_avg / opt_avg if opt_avg > 0 else 0
        label = label_map.get(phase, phase.capitalize())
        lines.append(f"  {label:<20} {bl_avg:>13.3f}s {opt_avg:>15.3f}s {sp:>7.2f}x")

    # Overall savings
    bl_total = _avg([qr.bl_total for qr in results])
    opt_total = _avg([qr.opt_total for qr in results])
    saved = bl_total - opt_total
    pct = saved / bl_total * 100 if bl_total > 0 else 0
    lines.append(f"  {'─' * 62}")
    lines.append(f"  Average time saved: {saved:.3f}s ({pct:.1f}% faster)")

    # Per-query table
    lines.append("")
    lines.append("=" * w)
    lines.append("  PER-QUERY BREAKDOWN")
    lines.append("=" * w)
    lines.append(f"  {'ID':>4} {'Baseline':>10} {'Optimized':>10} {'Speedup':>8}  Query")
    lines.append(f"  {'─' * 72}")
    for qr in results:
        sp = qr.total_speedup
        qtxt = qr.query_text[:40] + ("..." if len(qr.query_text) > 40 else "")
        lines.append(
            f"  {qr.query_id:>4} {qr.bl_total:>9.3f}s {qr.opt_total:>9.3f}s {sp:>7.2f}x  {qtxt}"
        )

    report = "\n".join(lines) + "\n"
    path = output_dir / "summary_report.txt"
    path.write_text(report, encoding="utf-8")
    # Also print to console
    print(report)
    print(f"  TXT  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_batch",
        description="Batch stress-test benchmarker for MCP workflow optimizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  uv run python benchmark_batch.py --queries queries.txt
  uv run python benchmark_batch.py --queries queries.txt --runs 5 --output-dir results/
  uv run python benchmark_batch.py --queries queries.txt --resume
""",
    )
    parser.add_argument(
        "--queries", required=True, metavar="FILE",
        help="Path to query file (one query per line, # comments ignored).",
    )
    parser.add_argument(
        "--runs", type=int, default=3, metavar="N",
        help="Number of repeats per query per mode (default: 3).",
    )
    parser.add_argument(
        "--model-id",
        default="watsonx/meta-llama/llama-3-3-70b-instruct",
        metavar="MODEL",
        help="LiteLLM model string (default: watsonx/meta-llama/llama-3-3-70b-instruct).",
    )
    parser.add_argument(
        "--cooldown", type=float, default=2.0, metavar="SECS",
        help="Seconds to wait between queries (default: 2.0).",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", metavar="DIR",
        help="Output directory for results (default: benchmark_results/).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run — skip queries already in raw_runs.json.",
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    queries = read_queries(args.queries)
    await run_benchmark(
        queries=queries,
        runs=args.runs,
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        cooldown=args.cooldown,
        resume=args.resume,
    )


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
