"""Teacher data generation, autonomous deployment and independent scoring."""

import numpy as np
from copy import deepcopy
from .host import Host, scheduled_fit
from .targets import Circuit, Physical, reference
from .domains import Domain


def rmse(x, y):
    return float(np.sqrt(np.mean(np.square(x - y))))


def signals(rng, n):
    return 0.48 * np.sin(
        np.arange(n) * rng.uniform(0.04, 0.32) + rng.uniform(-3, 3)
    ) + rng.normal(0, 0.26, n)


def train(target, seed, width=32):
    rng = np.random.default_rng(seed)
    drives = np.stack([signals(rng, 64) for _ in range(16)])
    paths = np.stack(
        [reference(target, u, rng.uniform(-0.4, 0.4, target.size)) for u in drives]
    )
    x, y = paths[:, :-1], paths[:, 1:]

    def fitted(linear=False):
        return Host(target.size, width, rng, linear).fit(
            x.reshape(-1, target.size), drives.ravel(), y.reshape(-1, target.size)
        )

    host, linear = fitted(), fitted(True)
    scheduled = deepcopy(host)
    scheduled_fit(scheduled, x, drives, y, rng)
    return host, linear, scheduled


def study(kind, seed, size=3):
    rng = np.random.default_rng(seed)
    if kind == "circuit":
        target = Circuit.random(size, rng)
    elif kind in Domain.sizes:
        target = Domain(kind)
    else:
        target = Physical(kind)
    host, linear, scheduled = train(target, seed + 101)
    drive = signals(np.random.default_rng(seed + 202), 256)
    drive[128:136] += 0.6
    initial = np.zeros(target.size)
    prediction = host.rollout(drive, initial)
    truth = reference(target, drive, initial)[1:]
    teacher = host.step(np.vstack((initial, truth[:-1])), drive)
    mixed = scheduled.rollout(drive, initial)
    results = dict(
        autonomous_rmse=rmse(prediction, truth),
        linear_rmse=rmse(linear.rollout(drive, initial), truth),
        initial_hold_rmse=rmse(np.zeros_like(truth), truth),
        teacher_rmse=rmse(teacher, truth),
        scheduled_rmse=rmse(mixed, truth),
        tail_rmse=rmse(prediction[-64:], truth[-64:]),
        mae=float(np.abs(prediction - truth).mean()),
        saturated_fraction=float(np.mean(np.abs(prediction) >= 8)),
        scheduled_saturation=float(np.mean(np.abs(mixed) >= 8)),
        model_bytes=host.readout.nbytes + host.projection.nbytes,
    )
    if kind == "lif":
        hits = (prediction[:, 1] >= 0.5) & (truth[:, 1] == 1)
        results.update(
            spike_precision=float(hits.sum() / max(1, (prediction[:, 1] >= 0.5).sum())),
            spike_recall=float(hits.sum() / max(1, truth[:, 1].sum())),
        )
    return results


def structural(seed):
    """Learn a local graph-conditioned rule, deploy on disjoint weighted graphs."""
    rng = np.random.default_rng(seed)
    inputs, outputs = [], []
    for _ in range(20):
        graph = Circuit.random(5, rng)
        drives = signals(rng, 48)
        path = reference(graph, drives)
        for state, drive, label in zip(path, drives, path[1:]):
            inputs.extend(
                np.column_stack((state, graph.weights @ state, graph.ports * drive))
            )
            outputs.extend(label)
    x = np.array(inputs)
    from .host import ridge

    proj = rng.normal(size=(4, 24))

    def features(raw):
        augmented = np.column_stack((raw, np.ones(len(raw))))
        return np.column_stack((augmented, np.tanh(augmented @ proj)))

    coef = ridge(features(x), np.array(outputs))
    graph = Circuit.random(10, rng)
    u = signals(rng, 256)

    def deploy(weights):
        state = np.zeros(10)
        trace = []
        for drive in u:
            state = (
                features(np.column_stack((state, weights @ state, graph.ports * drive)))
                @ coef
            )
            trace.append(state)
        return np.array(trace)

    truth = reference(graph, u)[1:]
    return dict(
        unseen_rmse=rmse(deploy(graph.weights), truth),
        shuffled_edges_rmse=rmse(deploy(graph.weights[::-1]), truth),
    )
