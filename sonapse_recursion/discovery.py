"""Bounded search for efficient host architectures and curriculum failure points."""

import numpy as np
from .emulation import signals, rmse, train
from .targets import Circuit, reference
from .host import Host


def curriculum(seed):
    rng = np.random.default_rng(seed)
    metrics = {}
    passed = 0
    for n in (1, 2, 5, 10, 25):
        graph = Circuit.random(n, rng)
        host, _, _ = train(graph, seed + 3, width=32)
        validation = signals(rng, 128)
        error = rmse(
            host.rollout(validation, np.zeros(n)), reference(graph, validation)[1:]
        )
        metrics["last_validation_rmse"] = error
        if error > 0.015:
            break
        passed = n
    test = signals(rng, 256)
    return dict(
        metrics,
        largest_passed=passed,
        last_size=n,
        last_test_rmse=rmse(
            host.rollout(test, np.zeros(n)), reference(graph, test)[1:]
        ),
    )


def pareto(seed):
    """Seeded population search with independent validation and final testing."""
    rng = np.random.default_rng(seed)
    target = Circuit.random(3, rng)
    train_u = signals(rng, 300)
    train_path = reference(target, train_u)
    val_u, test_u = signals(rng, 128), signals(rng, 256)
    val_y, test_y = reference(target, val_u)[1:], reference(target, test_u)[1:]

    def fit(genome):
        host = Host(3, 0, rng)
        host.projection = genome.copy()
        host.fit(train_path[:-1], train_u, train_path[1:])
        return host

    def score(genome):
        host = fit(genome)
        return (
            rmse(host.rollout(val_u, np.zeros(3)), val_y),
            int(genome.size + host.readout.size),
        )

    def front(pop):
        scores = [score(g) for g in pop]
        return [
            i
            for i, (err, params) in enumerate(scores)
            if not any(
                (e <= err and p <= params and (e < err or p < params))
                for e, p in scores
            )
        ]

    parents = [rng.normal(size=(5, w)) for w in (4, 8, 12, 16)]
    base = parents.copy()
    for _ in range(4):
        children = []
        for _ in range(5):
            child = parents[rng.integers(len(parents))].copy()
            op = rng.integers(3)
            if op == 0 and child.shape[1] < 32:
                child = np.column_stack((child, rng.normal(size=5)))
            elif op == 1 and child.shape[1] > 2:
                child = np.delete(child, rng.integers(child.shape[1]), axis=1)
            else:
                child[rng.integers(5), rng.integers(child.shape[1])] += rng.normal(
                    0, 0.4
                )
            children.append(child)
        population = parents + children
        parents = [population[i] for i in front(population)]

    def winner(pop):
        return min(pop, key=lambda g: score(g)[0] + 0.00001 * score(g)[1])

    best, prior = winner(parents), winner(base)
    return dict(
        front_size=len(parents),
        selected_width=best.shape[1],
        selected_test_rmse=rmse(fit(best).rollout(test_u, np.zeros(3)), test_y),
        initial_test_rmse=rmse(fit(prior).rollout(test_u, np.zeros(3)), test_y),
    )


def closed_loop(seed):
    """One-tick plant control via lossy delayed feedback with causal confidence gating."""
    rng = np.random.default_rng(seed)
    disturbance = rng.normal(0, 0.045, 384)
    disturbance[192:200] += 0.3
    losses = rng.random(384) < 0.15
    delays = rng.integers(2, 8, 384)
    out = {}
    for policy in ("hold", "predict", "confidence"):
        x, estimate, slope, confidence = 1.0, 0.0, 0.0, 1.0
        stamp = -1
        queue, states = {}, []
        for tick, noise in enumerate(disturbance):
            if not losses[tick]:
                queue.setdefault(tick + int(delays[tick]), []).append((tick, x))
            for sent, value in queue.pop(tick, []):
                if sent <= stamp:
                    continue
                dt = sent - stamp
                residual = abs(value - (estimate + slope * dt))
                confidence = 0.8 * confidence + 0.2 * np.exp(-4 * residual)
                slope = np.clip((value - estimate) / dt, -0.3, 0.3)
                estimate, stamp = value, sent
            gate = {"hold": 0, "predict": 1, "confidence": confidence}[policy]
            prediction = estimate + gate * slope * min(8, tick - stamp)
            x = 0.98 * x - 0.22 * np.clip(prediction, -2, 2) + noise
            states.append(x)
        out[policy + "_rms"] = rmse(np.array(states), 0)
        out[policy + "_peak"] = float(np.max(np.abs(states)))
    return out
