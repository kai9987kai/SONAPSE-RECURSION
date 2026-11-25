"""Two virtual clocks: distal child readout and independent parent task."""

import numpy as np
from .targets import Circuit
from .host import Host
from .emulation import train, signals, rmse


def hierarchy(seed):
    rng = np.random.default_rng(seed)
    target = Circuit.random(2, rng)
    inner, _, _ = train(target, seed + 71)

    def path(u, learned):
        state = np.zeros(4)
        states = [state]
        for drive in u:
            native, clock = state[:2]
            child = state[2:]
            if clock < 0.5:
                child = (inner if learned else target).step(child, drive)
            state = np.r_[
                0.6 * native + 0.4 * np.tanh(child[0] + drive), 1 - clock, child
            ]
            states.append(state)
        return np.array(states)

    drives = np.stack([signals(rng, 64) for _ in range(24)])
    trial = signals(rng, 256)
    truth = path(trial, False)[1:]
    out = {}
    for label, learned in (("distilled", True), ("flat", False)):
        teacher = np.stack([path(u, learned) for u in drives])
        # Identical random projections and example count in both training conditions.
        host = Host(4, 64, np.random.default_rng(seed + 72))
        host.fit(
            teacher[:, :-1].reshape(-1, 4),
            drives.ravel(),
            teacher[:, 1:].reshape(-1, 4),
        )
        rollout = host.rollout(trial, np.zeros(4))
        out[label + "_child_rmse"] = rmse(rollout[:, 2:], truth[:, 2:])
        out[label + "_native_rmse"] = rmse(rollout[:, 0], truth[:, 0])
        out[label + "_clock_accuracy"] = float(
            np.mean((rollout[:, 1] > 0.5) == truth[:, 1])
        )
    return out
