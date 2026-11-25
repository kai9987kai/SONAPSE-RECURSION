"""Finite host state bank with hierarchical leases and immediate interruption."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Lease:
    start: int
    stop: int
    parent: str | None
    active: bool = True


class Bank:
    def __init__(self, cells=64, reserve=8, max_depth=3):
        if not 0 <= reserve <= cells or max_depth < 1:
            raise ValueError("Invalid reservation")
        self.state = np.zeros(cells)
        self.reserve, self.max_depth = reserve, max_depth
        self.leases = {}

    def allocate(self, name, cells, parent=None):
        if (
            not isinstance(name, str)
            or not name
            or name in self.leases
            or type(cells) is not int
            or cells <= 0
        ):
            raise ValueError("Invalid graft")
        ancestor, depth = parent, 1
        while ancestor is not None:
            depth += 1
            ancestor = self.leases[ancestor].parent
        if depth > self.max_depth:
            raise ValueError("Recursion limit")
        low, high = self.reserve, len(self.state)
        if parent is not None:
            low, high = self.leases[parent].start, self.leases[parent].stop
        siblings = sorted(
            (g.start, g.stop) for g in self.leases.values() if g.parent == parent
        )
        for start, stop in siblings + [(high, high)]:
            if start - low >= cells:
                self.leases[name] = Lease(low, low + cells, parent)
                return
            low = stop
        raise MemoryError("Host state reservation exhausted")

    def view(self, name):
        lease = self.leases[name]
        return self.state[lease.start : lease.stop]

    def enabled(self, name):
        while name is not None:
            lease = self.leases[name]
            if not lease.active:
                return False
            name = lease.parent
        return True

    def suspend(self, name, paused=True):
        self.leases[name].active = not paused

    def remove(self, name):
        for child in [n for n, g in self.leases.items() if g.parent == name]:
            self.remove(child)
        self.view(name).fill(0)
        del self.leases[name]
