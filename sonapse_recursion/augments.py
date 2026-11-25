"""Controlled external augmentation, interruption, and programmable sequential logic."""

import numpy as np
from .emulation import rmse, signals, train
from .targets import Circuit
from .host import ridge
from .resources import Bank


def external(seed):
    rng = np.random.default_rng(seed)
    u = signals(rng, 800)
    target = np.roll(u, 4)
    w = rng.normal(0, 0.19, (16, 16))
    g = rng.normal(size=16)

    def features(size):
        state = np.zeros(size)
        x = []
        for drive in u:
            state = np.tanh(w[:size, :size] @ state + g[:size] * drive)
            x.append(state.copy())
        return np.array(x)

    native, aux = features(4), features(16)
    arms = {
        "native": native,
        "external_delayed": np.column_stack((native, np.roll(aux, 2, axis=0))),
        "colocated": np.column_stack((native, aux)),
    }
    out = {}
    for name, x in arms.items():
        coef = ridge(x[4:400], target[4:400])
        out[name + "_rmse"] = rmse(x[400:] @ coef, target[400:])
    return dict(
        out, host_cells=4, external_cells=16, port_values_per_tick=16, external_delay=2
    )


def interruption(seed):
    rng = np.random.default_rng(seed)
    host, _, _ = train(Circuit.random(2, rng), seed + 9)
    u = signals(rng, 64)
    bank = Bank(8, 2)
    bank.allocate("sim", 2)
    state = bank.view("sim")
    for drive in u[:32]:
        state[:] = host.step(state, drive)
    snapshot = state.copy()
    bank.suspend("sim")
    for drive in u[32:40]:
        if bank.enabled("sim"):
            state[:] = host.step(state, drive)
    drift = rmse(snapshot, state)
    bank.suspend("sim", False)
    for drive in u[32:]:
        state[:] = host.step(state, drive)
    return dict(
        paused_drift=drift,
        resume_rmse=rmse(state, host.rollout(u, np.zeros(2))[-1]),
        protected_cells=2,
        paused_wall_ticks=8,
        virtual_ticks=64,
    )


def machine(seed):
    """Condition a learned local multiplexer on an unseen binary state table."""
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(7, 64))

    def encode(rows):
        x = np.column_stack((rows, np.ones(len(rows))))
        return np.column_stack((x, np.tanh(x @ projection)))

    tables = rng.permutation(16)

    def bits(table):
        return [(int(table) >> i) & 1 for i in range(4)]

    rows, labels = [], []
    for table in tables[:12]:
        rule = bits(table)
        for state in (0, 1):
            for drive in (0, 1):
                rows.append([state, drive, *rule])
                labels.append(rule[2 * state + drive])
    coef = ridge(encode(rows), np.array(labels))
    scores = []
    for table in tables[12:]:
        rule = bits(table)
        actual, expected = 0, 0
        for drive in rng.integers(0, 2, 128):
            actual = int((encode([[actual, drive, *rule]]) @ coef)[0] > 0.5)
            expected = rule[2 * expected + drive]
            scores.append(actual == expected)
    return dict(
        unseen_state_accuracy=float(np.mean(scores)),
        training_programs=12,
        held_out_programs=4,
    )
