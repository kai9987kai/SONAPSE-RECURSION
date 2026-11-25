"""Fixed-width nonlinear neural transition with a trained recurrent readout."""

import numpy as np


def ridge(x, y, penalty=1e-5):
    return np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


class Host:
    def __init__(self, size, width, rng, linear=False):
        self.size = size
        self.projection = rng.normal(0, 0.8, (size + 2, 0 if linear else width))
        self.readout = None

    def features(self, state, drive):
        raw = np.concatenate(
            (state, np.asarray(drive)[..., None], np.ones((*state.shape[:-1], 1))),
            axis=-1,
        )
        return np.concatenate((raw, np.tanh(raw @ self.projection)), axis=-1)

    def fit(self, states, drives, following):
        self.readout = ridge(self.features(states, drives), following)
        return self

    def step(self, state, drive):
        # Finite activation range shared by every deployed host and linear control.
        return np.clip(self.features(state, drive) @ self.readout, -8.0, 8.0)

    def rollout(self, inputs, initial, perturb=None):
        state = np.array(initial, copy=True)
        trace = []
        for tick, drive in enumerate(inputs):
            state = self.step(state, drive)
            if perturb is not None:
                state = state + perturb[tick]
            trace.append(state)
        return np.asarray(trace)


def scheduled_fit(host, states, drives, following, rng):
    """Refit on model-generated contexts paired with time-aligned teacher labels."""
    contexts = []
    for start, sequence, truth in zip(states, drives, following):
        context = start[0]
        for teacher, drive, label in zip(start, sequence, truth):
            if rng.random() < 0.5:
                context = teacher
            contexts.append(context)
            context = host.step(context, drive)
    host.fit(np.asarray(contexts), drives.ravel(), following.reshape(-1, host.size))
