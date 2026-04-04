"""Parallel executor subclass — runs independent plan steps concurrently.

Uses ``Plan.dependency_layers()`` to group steps by topological level,
then ``asyncio.gather()`` to execute each layer in parallel while
preserving dependency ordering between layers.

Opt-in via the ``--parallel`` CLI flag.
"""

from __future__ import annotations

import asyncio
import logging

from .executor import Executor
from .models import Plan, StepResult

_log = logging.getLogger(__name__)


class ParallelExecutor(Executor):
    """Executor subclass that runs independent plan steps concurrently.

    Steps are grouped into *dependency layers* using Kahn's algorithm.
    All steps within a layer execute concurrently via ``asyncio.gather()``.
    The next layer starts only after the current layer has fully completed
    and all results have been recorded in the shared context dict.

    Error handling is *fail-tolerant*: if one step in a layer fails,
    sibling steps still run to completion. Downstream steps that depend
    on a failed step will fail naturally at arg-resolution time.
    """

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        """Execute plan steps layer-by-layer, gathering independent steps."""
        layers = plan.dependency_layers()
        total = sum(len(layer) for layer in layers)
        context: dict[int, StepResult] = {}
        results: list[StepResult] = []

        # Pre-fetch tool schemas exactly like the sequential Executor does, so
        # the LLM receives precise parameter names when resolving arguments.
        server_names = {step.server for layer in layers for step in layer}
        tool_schemas: dict[str, dict[str, str]] = {}
        for name in server_names:
            path = self._server_paths.get(name)
            if path is None:
                continue
            try:
                from .executor import _list_tools  # re-use the module-level helper
                tools = await _list_tools(path)
                tool_schemas[name] = {
                    t["name"]: ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    for t in tools
                }
            except Exception:  # noqa: BLE001
                tool_schemas[name] = {}

        for layer_idx, layer in enumerate(layers):
            _log.info(
                "Executing layer %d/%d (%d step(s) in parallel)",
                layer_idx + 1,
                len(layers),
                len(layer),
            )

            # All steps in this layer have their dependencies satisfied.
            layer_results = await asyncio.gather(*[
                self.execute_step(
                    step,
                    context,
                    question,
                    tool_schema=tool_schemas.get(step.server, {}).get(step.tool, ""),
                )
                for step in layer
            ])

            for result in layer_results:
                if result.success:
                    _log.info("Step %d/%d OK.", result.step_number, total)
                else:
                    _log.warning(
                        "Step %d/%d FAILED: %s",
                        result.step_number,
                        total,
                        result.error,
                    )
                context[result.step_number] = result
                results.append(result)

        return results
