"""Multi-resource dispatch with protected reservations and preemptible quanta."""

from dataclasses import dataclass, fields
import numpy as np


@dataclass(frozen=True)
class Budget:
    ops: int
    bytes: int
    bandwidth: int
    energy: int
    attention: int

    def __post_init__(self):
        if any(
            type(getattr(self, f.name)) is not int or getattr(self, f.name) < 0
            for f in fields(self)
        ):
            raise ValueError("Budgets require nonnegative integer units")

    def fits(self, available):
        return all(
            getattr(self, f.name) <= getattr(available, f.name) for f in fields(self)
        )

    def minus(self, cost):
        return Budget(
            *(getattr(self, f.name) - getattr(cost, f.name) for f in fields(self))
        )


@dataclass
class Job:
    name: str
    cost: Budget
    remaining: int
    priority: int
    deadline: int
    value: float
    native: bool = False


class Governor:
    def __init__(self, capacity, reserve):
        if not reserve.fits(capacity):
            raise ValueError("Reservation exceeds capacity")
        self.capacity, self.reserve = capacity, reserve

    def dispatch(self, jobs, tick, policy="deadline", emergency=False):
        keys = {
            "fifo": lambda j: 0,
            "priority": lambda j: -j.priority,
            "deadline": lambda j: j.deadline,
            "utility": lambda j: -j.value / max(1, j.cost.energy),
        }
        if policy not in keys:
            raise ValueError("Unknown scheduling policy")
        available = self.capacity
        pending = self.reserve
        ran = []
        for job in sorted(jobs, key=keys[policy]):
            if (
                job.remaining <= 0
                or tick > job.deadline
                or (emergency and not job.native)
            ):
                continue
            allowance = available if job.native else available.minus(pending)
            if job.cost.fits(allowance):
                job.remaining -= 1
                available = available.minus(job.cost)
                if job.native:
                    pending = Budget(
                        *(
                            max(0, getattr(pending, f.name) - getattr(job.cost, f.name))
                            for f in fields(pending)
                        )
                    )
                ran.append(job.name)
        return ran


def governance(seed):
    rng = np.random.default_rng(seed)
    templates = [
        (int(rng.integers(1, 4)), int(rng.integers(3, 12)), float(rng.random()))
        for _ in range(16)
    ]
    result = {}
    for policy in ("fifo", "priority", "deadline", "utility"):
        jobs = [
            Job(
                str(i),
                Budget(2, 4, 1, 1, 1),
                n,
                round(value * 5),
                deadline,
                value,
                i % 4 == 0,
            )
            for i, (n, deadline, value) in enumerate(templates)
        ]
        governor = Governor(Budget(10, 20, 5, 5, 5), Budget(2, 4, 1, 1, 1))
        for tick in range(12):
            governor.dispatch(jobs, tick, policy, emergency=tick == 4)
        result[policy + "_completed"] = sum(j.remaining == 0 for j in jobs)
        result[policy + "_utility"] = sum(j.value for j in jobs if j.remaining == 0)
    return result
