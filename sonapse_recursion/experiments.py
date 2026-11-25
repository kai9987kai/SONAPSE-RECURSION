"""Paired synthetic interventions; each experiment receives an independent seed."""

import numpy as np
from .emulation import train, signals, rmse
from .targets import Circuit, reference
from .host import ridge
from .resources import Bank
from .grafts import Grafts, Specification


def graft(seed):
    rng = np.random.default_rng(seed)
    target = Circuit.random(2, rng)
    host, _, _ = train(target, seed + 12)
    bank = Bank(40, 8)
    manager = Grafts(bank)
    u = signals(rng, 768)
    ref = reference(target, u)[1:, 0]
    # Same encoder dimensions, downstream labels, and readout fit in all arms.
    estimates = {}
    decoy = Circuit.random(2, np.random.default_rng(seed + 77))
    control, _, _ = train(decoy, seed + 12)
    for arm in ("trained", "other_target", "random", "disconnected"):
        manager.install(
            Specification("memory"), control if arm == "other_target" else host
        )
        state = bank.view("memory")
        rows = []
        for tick, drive in enumerate(u):
            if arm == "random":
                state[:] = np.tanh(0.55 * state + np.array([0.3, -0.8]) * drive)
            else:
                manager.communicate("memory", drive, tick)
            rows.append(
                [drive, *(state if arm != "disconnected" else np.zeros(2)), 1.0]
            )
        x = np.array(rows)
        coef = ridge(x[:384], ref[:384])
        estimates[arm + "_rmse"] = rmse(x[384:] @ coef, ref[384:])
        if arm == "trained":
            estimates["trained_inference_macs"] = manager.multiply_adds
            estimates["graft_parameters"] = manager.describe("memory")["weights"]
        manager.remove("memory")
    return estimates


def nested(seed):
    """Outer neural emulator distils the state transition of a trained inner host."""
    rng = np.random.default_rng(seed)
    target = Circuit.random(2, rng)
    inner, _, _ = train(target, seed + 13)
    outer, _, _ = train(inner, seed + 14)
    u = signals(rng, 256)
    bank = Bank(16, 4)
    bank.allocate("outer", 8)
    bank.allocate("child", 2, "outer")
    state = bank.view("child")
    path = []
    for drive in u:
        state[:] = outer.step(state, drive)
        path.append(state.copy())
    truth = reference(target, u)[1:]
    child = inner.rollout(u, np.zeros(2))
    return dict(
        inner_rmse=rmse(child, truth),
        outer_rmse=rmse(np.array(path), truth),
        distillation_rmse=rmse(np.array(path), child),
        allocated_cells=8,
        child_cells=2,
    )


def interference(seed):
    rng = np.random.default_rng(seed)
    target = Circuit.random(3, rng)
    host, _, _ = train(target, seed + 22)
    u = signals(rng, 256)
    truth = reference(target, u)[1:]
    ideas = rng.normal(0, 0.09, (256, 3))
    return {
        arm + "_rmse": rmse(host.rollout(u, np.zeros(3), ideas * gate), truth)
        for arm, gate in [("shared", 1), ("partial", 0.5), ("isolated", 0)]
    }


def survival(seed):
    """Grid forager with shared exogenous weather, food sites and threat schedule."""
    rng = np.random.default_rng(seed)
    threats = rng.random((48, 160)) < 0.1
    sites = rng.integers(-4, 5, (48, 160))
    out = {}
    for policy in ("native", "busy", "interrupt", "protected"):
        durations, progress, interrupts = [], 0, 0
        for hazards, foods in zip(threats, sites):
            health, x, jobs = 6, 0, 0
            for tick, (threat, food) in enumerate(zip(hazards, foods), 1):
                if tick % 7 == 1 and policy != "native":
                    jobs = 4
                paused = policy == "interrupt" and threat and jobs > 0
                interrupts += paused
                compute = min(jobs, 1) if not paused else 0
                jobs -= compute
                progress += compute
                attentive = not compute or policy == "protected"
                if threat:
                    health -= 0 if attentive else 2
                elif attentive:
                    x += int(np.sign(food - x))
                    if x == food:
                        health = min(6, health + 1)
                if health <= 0:
                    break
            durations.append(tick)
        out.update(
            {
                policy + "_ticks": float(np.mean(durations)),
                policy + "_work": progress,
                policy + "_interrupts": int(interrupts),
            }
        )
    return out


def channel(seed, loss_rate=0.12, min_delay=2, max_delay=6):
    if not 0 <= loss_rate <= 1 or min_delay < 0 or max_delay < min_delay:
        raise ValueError("Invalid channel")
    rng = np.random.default_rng(seed)
    u = signals(rng, 640)
    truth = np.sin(np.arange(640) * 0.11) + 0.12 * u
    arrivals = {}
    lost = 0
    for tick, value in enumerate(truth):
        if rng.random() < loss_rate:
            lost += 1
        else:
            delay = int(rng.integers(min_delay, max_delay + 1))
            arrivals.setdefault(tick + delay, []).append((tick, value))
    coef = ridge(np.column_stack((truth[1:319], truth[:318])), truth[2:320])
    previous, latest = (-1, 0.0), (-1, 0.0)
    hold, pred = [], []
    for tick in range(640):
        for stamp, value in sorted(arrivals.get(tick, [])):
            if stamp > latest[0]:
                previous, latest = latest, (stamp, value)
        estimate = latest[1]
        if latest[0] - previous[0] == 1:
            old = previous[1]
            for _ in range(min(8, tick - latest[0])):
                old, estimate = estimate, np.clip(coef @ [estimate, old], -1.3, 1.3)
        hold.append(latest[1])
        pred.append(estimate)
    return dict(
        hold_rmse=rmse(np.array(hold[320:]), truth[320:]),
        predictive_rmse=rmse(np.array(pred[320:]), truth[320:]),
        packets_lost=lost,
        packets_sent=640,
    )


def adaptation(seed):
    rng = np.random.default_rng(seed)
    # Bidirectional scalar reconstruction: each peer learns encoder and decoder.
    ea, eb, da, db = rng.uniform(0.2, 0.7, 4)
    initial = np.array([ea, eb, da, db])
    examples = rng.uniform(-1, 1, (600, 2))
    test = rng.uniform(-1, 1, (200, 2))
    out = {}
    for arm in ("frozen", "decoder_only", "reciprocal"):
        ea, eb, da, db = initial
        for x, y in examples:
            a, b = np.tanh(ea * x), np.tanh(eb * y)
            ex, ey = x - db * a, y - da * b
            if arm == "reciprocal":
                ea += 0.03 * ex * db * (1 - a * a) * x
                eb += 0.03 * ey * da * (1 - b * b) * y
            if arm != "frozen":
                db += 0.03 * ex * a
                da += 0.03 * ey * b
        x, y = test.T
        out[arm + "_rmse"] = rmse(
            np.stack((db * np.tanh(ea * x), da * np.tanh(eb * y))), test.T
        )
    return out


def evolution(seed, history=None):
    rng = np.random.default_rng(seed)
    sequence = signals(rng, 500)
    labels = np.roll(sequence, 3)

    def evaluate(w, start, end):
        state = np.zeros(len(w))
        states = []
        for u in sequence:
            state = np.tanh(w @ state + gains * u)
            states.append(state)
        x = np.array(states)
        readout = ridge(x[4:240], labels[4:240])
        return rmse(x[start:end] @ readout, labels[start:end])

    gains = rng.normal(size=6)
    parent = rng.normal(0, 0.25, (6, 6))
    baseline = parent.copy()
    score = evaluate(parent, 240, 370)
    accepted = 0
    search_parent = baseline.copy()
    search_score = score
    for generation in range(20):
        child = parent.copy()
        i, j = rng.integers(0, 6, 2)
        prune = rng.random() < 0.2
        delta = 0 if prune else rng.normal(0, 0.35)
        child[i, j] = 0 if prune else child[i, j] + delta
        fixed_child = baseline.copy()
        fixed_child[i, j] = 0 if prune else fixed_child[i, j] + delta
        control_score = evaluate(fixed_child, 240, 370)
        if control_score < search_score:
            search_score, search_parent = control_score, fixed_child
        candidate = evaluate(child, 240, 370)
        if candidate < score:
            parent, score = child, candidate
            accepted += 1
        if history is not None:
            history.append(
                dict(
                    generation=generation + 1,
                    score=score,
                    candidate=candidate,
                    independent=search_score,
                    accepted=accepted,
                    edges=int(np.count_nonzero(parent)),
                )
            )
    return dict(
        fixed_test_rmse=evaluate(baseline, 370, 500),
        evolved_test_rmse=evaluate(parent, 370, 500),
        independent_search_rmse=evaluate(search_parent, 370, 500),
        accepted=accepted,
        active_edges=int(np.count_nonzero(parent)),
    )


def logic(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(3, 16))

    def features(bits):
        x = np.column_stack((bits, np.ones(len(bits))))
        return np.column_stack((x, np.tanh(x @ w)))

    def labels(bits):
        a, b = bits.T
        return np.column_stack((a & b, a | b, 1 - a, a ^ b))

    bits = rng.integers(0, 2, (80, 2))
    coef = ridge(features(bits), labels(bits))
    cases = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    return dict(
        truth_table_accuracy=float(
            np.mean((features(cases) @ coef > 0.5) == labels(cases))
        )
    )
