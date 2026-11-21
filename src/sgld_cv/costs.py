"""Measured wall times and actual operation counts, with no FLOP equivalences.

An observation gradient is one likelihood-gradient contribution at one theta.
Curvature operations and data row visits are counted separately. Row visits
are logical operations, not measured physical memory loads.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager
from time import perf_counter


@dataclass
class CostLedger:
    seconds: dict[str, float] = field(default_factory=dict)
    operations: dict[str, int] = field(default_factory=dict)
    phase_operations: dict[str, dict[str, int]] = field(default_factory=dict)

    def count(self, name: str, number: int = 1) -> None:
        if not isinstance(number, int) or number < 0:
            raise ValueError("operation counts must be nonnegative integers")
        self.operations[name] = self.operations.get(name, 0) + int(number)

    @contextmanager
    def measure(self, phase: str):
        start = perf_counter()
        before = dict(self.operations)
        try:
            yield
        finally:
            self.seconds[phase] = self.seconds.get(phase, 0.0) + perf_counter() - start
            attributed = self.phase_operations.setdefault(phase, {})
            for name, value in self.operations.items():
                delta = value - before.get(name, 0)
                if delta:
                    attributed[name] = attributed.get(name, 0) + delta

    def copy(self):
        return CostLedger(dict(self.seconds), dict(self.operations),
                          {p: dict(v) for p, v in self.phase_operations.items()})

    def add(self, other):
        for key, value in other.seconds.items():
            self.seconds[key] = self.seconds.get(key, 0.0) + value
        for key, value in other.operations.items():
            self.count(key, value)
        for phase, counts in other.phase_operations.items():
            target = self.phase_operations.setdefault(phase, {})
            for name, value in counts.items():
                target[name] = target.get(name, 0) + value

    def snapshot(self):
        return {"seconds": dict(self.seconds), "operations": dict(self.operations),
                "phase_operations": {p: dict(v) for p, v in self.phase_operations.items()},
                "total_seconds": sum(self.seconds.values())}
