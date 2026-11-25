"""Sparse event propagation for a discrete threshold connectome."""

from collections import defaultdict
from heapq import heappop, heappush
import numpy as np


def sparse_study(seed, n=4096, ticks=256, inputs=8):
    rng = np.random.default_rng(seed)
    edges = [(node, (node + 1) % n, 1.0, 1) for node in range(n)]
    edges.extend((node, int(rng.integers(n)), 0.25, 3) for node in range(n))
    substrate = EventConnectome(n, edges)
    for node in rng.choice(n, inputs, replace=False):
        substrate.inject(0, int(node), 1.0)
    events = substrate.advance(ticks)
    return dict(
        nodes=n,
        edges=len(edges),
        spikes=len(events),
        event_operations=substrate.operations,
        dense_node_ticks=n * ticks,
        active_nodes=len({node for _, node in events}),
    )


class EventConnectome:
    def __init__(self, size, edges, threshold=1.0, refractory=1):
        if type(size) is not int or size < 0 or threshold <= 0 or refractory < 0:
            raise ValueError("Invalid event substrate")
        self.size, self.threshold, self.refractory = size, threshold, refractory
        self.outgoing = defaultdict(list)
        for src, dst, weight, delay in edges:
            if not (
                0 <= src < size
                and 0 <= dst < size
                and type(delay) is int
                and delay >= 1
                and np.isfinite(weight)
            ):
                raise ValueError("Invalid edge")
            self.outgoing[src].append((dst, weight, delay))
        self.events = defaultdict(dict)
        self._times = []
        self.available = np.zeros(size, dtype=int)
        self.clock, self.operations = 0, 0

    def inject(self, time, node, value):
        if time < self.clock or not 0 <= node < self.size or not np.isfinite(value):
            raise ValueError("Invalid event")
        if time not in self.events:
            heappush(self._times, time)
        pending = self.events[time]
        pending[node] = pending.get(node, 0.0) + value

    def advance(self, until):
        if until < self.clock:
            raise ValueError("Cannot reverse time")
        trace = []
        while self._times:
            tick = self._times[0]
            if tick >= until:
                break
            heappop(self._times)
            pending = self.events.pop(tick)
            self.clock = tick
            for node, value in sorted(pending.items()):
                self.operations += 1
                if value >= self.threshold and tick >= self.available[node]:
                    self.available[node] = tick + 1 + self.refractory
                    trace.append((tick, node))
                    for dst, weight, delay in self.outgoing.get(node, ()):
                        self.inject(tick + delay, dst, weight)
                        self.operations += 1
        self.clock = until
        return trace
