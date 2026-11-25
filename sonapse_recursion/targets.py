"""Conventional reference dynamics. Never imported by the learned host."""

import numpy as np


class Circuit:
    def __init__(self, weights, ports):
        self.weights = np.asarray(weights, dtype=float)
        self.ports = np.asarray(ports, dtype=float)
        self.size = len(self.ports)
        if self.ports.shape != (self.size,) or self.weights.shape != (
            self.size,
            self.size,
        ):
            raise ValueError(
                "Expected square connectivity and one input gain per neuron"
            )
        if not all(np.isfinite(x).all() for x in (self.weights, self.ports)):
            raise ValueError("Non-finite circuit")

    @classmethod
    def random(cls, size, rng):
        w = rng.normal(size=(size, size)) * (rng.random((size, size)) < 0.55)
        w /= max(1, np.linalg.norm(w)) / 0.85
        return cls(w, rng.uniform(0.3, 1.1, size))

    def step(self, state, drive):
        return 0.4 * state + 0.6 * np.tanh(self.weights @ state + self.ports * drive)


class Physical:
    size = 2

    def __init__(self, kind):
        if kind not in (
            "pendulum",
            "spring",
            "collision",
            "rc",
            "flow",
            "lif",
            "plastic",
        ):
            raise ValueError(kind)
        self.kind = kind

    def step(self, state, drive):
        x, v = state
        if self.kind == "lif":
            voltage = 0.84 * x + 0.5 * drive + 0.15
            return np.array([0 if voltage >= 0.65 else voltage, float(voltage >= 0.65)])
        if self.kind == "plastic":
            activity = np.tanh(v * x + drive)
            return np.array([activity, np.clip(0.98 * v + 0.06 * x * activity, -1, 1)])
        if self.kind == "rc":
            return np.array([x + 0.12 * (drive - x), v + 0.08 * (x - v)])
        if self.kind == "flow":
            flux = 0.08 * (x - v)
            return np.array([x + 0.05 * (drive - x) - flux, 0.97 * v + flux])
        restoring = np.sin(x) if self.kind == "pendulum" else x
        v += 0.12 * (drive - restoring - 0.25 * v)
        x += 0.12 * v
        if self.kind == "collision" and abs(x) > 0.3:
            x, v = np.sign(x) * 0.3, -v * 0.8
        return np.array([x, v])


def reference(target, inputs, initial=None):
    state = np.zeros(target.size) if initial is None else np.array(initial, copy=True)
    trace = [state]
    for drive in inputs:
        state = target.step(state, drive)
        trace.append(state)
    return np.stack(trace)
