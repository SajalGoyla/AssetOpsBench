"""Data models for the plan-execute orchestration client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    step_number: int
    task: str
    server: str
    tool: str
    tool_args: dict
    dependencies: list[int]
    expected_output: str


@dataclass
class Plan:
    """An execution plan composed of ordered steps."""

    steps: list[PlanStep]
    raw: str  # Raw LLM output, preserved for debugging

    def get_step(self, number: int) -> Optional[PlanStep]:
        return next((s for s in self.steps if s.step_number == number), None)

    def resolved_order(self) -> list[PlanStep]:
        """Return steps in topological order (dependencies before dependents)."""
        seen: set[int] = set()
        ordered: list[PlanStep] = []

        def visit(n: int) -> None:
            if n in seen:
                return
            step = self.get_step(n)
            if step is None:
                return
            for dep in step.dependencies:
                visit(dep)
            seen.add(n)
            ordered.append(step)

        for step in self.steps:
            visit(step.step_number)
        return ordered

    def dependency_layers(self) -> list[list["PlanStep"]]:
        """Group steps into dependency layers for parallel execution.

        Layer 0: steps with no dependencies (can all run in parallel)
        Layer 1: steps whose dependencies are all in layer 0
        Layer N: steps whose dependencies are all in layers 0..N-1

        Returns:
            List of layers, where each layer is a list of PlanSteps
            that can execute concurrently.
        """
        if not self.steps:
            return []

        step_map = {s.step_number: s for s in self.steps}
        in_degree = {s.step_number: 0 for s in self.steps}
        dependents: dict[int, list[int]] = {s.step_number: [] for s in self.steps}

        for s in self.steps:
            for dep in s.dependencies:
                if dep in step_map:
                    in_degree[s.step_number] += 1
                    dependents[dep].append(s.step_number)

        layers: list[list[PlanStep]] = []
        ready = [n for n, deg in in_degree.items() if deg == 0]

        while ready:
            layer = [step_map[n] for n in sorted(ready)]
            layers.append(layer)
            next_ready: list[int] = []
            for n in ready:
                for dep_n in dependents[n]:
                    in_degree[dep_n] -= 1
                    if in_degree[dep_n] == 0:
                        next_ready.append(dep_n)
            ready = next_ready

        return layers


@dataclass
class StepResult:
    """Result of executing a single plan step."""

    step_number: int
    task: str
    server: str
    response: str
    error: Optional[str] = None
    tool: str = ""
    tool_args: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None
