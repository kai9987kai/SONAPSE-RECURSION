"""Creative proposals, scored in a separate forced oscillator simulator."""

import numpy as np


def design(seed):
    rng = np.random.default_rng(seed)
    train_drives = rng.normal(size=(6, 96))
    test_drives = rng.normal(size=(6, 96))
    initial = np.array([0.5, 0.15])

    def score(pair, drives):
        stiffness, damping = pair
        x, v = np.zeros(len(drives)), np.zeros(len(drives))
        total = 0.0
        for force in drives.T:
            v += 0.1 * (force - stiffness * x - damping * v)
            x += 0.1 * v
            total += np.mean(x * x + 0.02 * (damping * v) ** 2)
        return float(total / drives.shape[1])

    current = initial.copy()
    parents = [current.copy()]
    errors = [score(current, train_drives)]
    for _ in range(32):
        donor = parents[rng.integers(len(parents))]
        candidate = np.clip(
            0.7 * current + 0.3 * donor + rng.normal(0, 0.12, 2), 0.03, 2
        )
        loss = score(candidate, train_drives)
        if loss < errors[-1]:
            current = candidate
            parents.append(candidate.copy())
            errors.append(loss)
    return dict(
        initial_test_cost=score(initial, test_drives),
        selected_test_cost=score(current, test_drives),
        accepted=len(parents) - 1,
        proposals=32,
        stiffness=current[0],
        damping=current[1],
    )
