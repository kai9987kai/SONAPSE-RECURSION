"""Configurable host substrate. All model parameters use simulation units."""

import numpy as np
from .targets import Circuit


class Substrate:
    models = (
        "rate",
        "lif",
        "adex",
        "izhikevich",
        "compartment",
        "two_soma",
        "dendritic",
        "hybrid",
    )

    def __init__(self, graph: Circuit, model="rate", delay=1, plastic=False):
        if model not in self.models or not isinstance(delay, int) or delay < 1:
            raise ValueError("Unknown neuron model or nonpositive delay")
        self.model, self.plastic = model, plastic
        self.weights, self.ports = graph.weights.copy(), graph.ports.copy()
        n = graph.size
        self.v = np.full(n, -65.0) if model == "izhikevich" else np.zeros(n)
        self.aux = 0.2 * self.v
        self.spikes = np.zeros(n, dtype=bool)
        self.cooldown = np.zeros(n, dtype=int)
        self.history = np.zeros((delay, n))
        self.tick, self.energy = 0, 0.0

    def step(self, drive, modulation=0.0):
        if not np.isfinite([drive, modulation]).all():
            raise ValueError("Non-finite stimulus")
        slot = self.tick % len(self.history)
        previous = self.history[slot].copy()
        current = self.weights @ previous + self.ports * drive
        v = self.v
        model = self.model
        if model == "rate":
            self.v = 0.4 * v + 0.6 * np.tanh(current)
        elif model in ("compartment", "two_soma", "dendritic"):
            coupling = {"two_soma": 0.35, "compartment": 0.15, "dendritic": 0.22}[model]
            branch = np.tanh(2 * current) if model == "dendritic" else current
            self.aux += 0.18 * (branch - self.aux + coupling * (v - self.aux))
            self.v = v + 0.2 * (-v + coupling * (self.aux - v) + current)
        elif model == "izhikevich":
            self.v = v + 0.25 * (0.04 * v * v + 5 * v + 140 - self.aux + 10 * current)
            self.aux += 0.005 * (0.2 * v - self.aux)
        else:
            active = self.cooldown == 0
            self.cooldown = np.maximum(0, self.cooldown - 1)
            boost = (
                0.05 * np.exp(np.clip((v - 0.7) / 0.12, -20, 5))
                if model == "adex"
                else 0
            )
            self.v = np.where(active, 0.86 * v + current + boost - self.aux, 0.0)
            self.aux *= 0.94
        spiking = model in ("lif", "adex", "izhikevich", "hybrid")
        self.spikes = (self.v >= (30 if model == "izhikevich" else 1)) & spiking
        if spiking:
            self.v[self.spikes] = -65 if model == "izhikevich" else 0
            self.cooldown[self.spikes] = 2
            self.aux[self.spikes] += {"adex": 0.12, "izhikevich": 8}.get(model, 0)
        output = self.spikes.astype(float) if spiking else self.v.copy()
        if model == "hybrid":
            output += 0.2 * np.tanh(self.v)
        if self.plastic:
            self.weights += 0.001 * modulation * np.outer(output, previous)
            np.clip(self.weights, -2, 2, out=self.weights)
        self.history[slot] = output
        self.energy += len(v) + float(np.abs(output).sum())
        self.tick += 1
        return output.copy()
