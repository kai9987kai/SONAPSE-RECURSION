"""Paired crossing decisions with a constructive benefit and a cost of deliberation."""

import numpy as np


def river_load(flow, slope, material):
    """Peak of a lumped pressure-response model over a twelve-tick crossing."""
    incoming = np.maximum(0.0, flow + slope * np.arange(12))
    pressure = 0.0
    peak = 0.0
    for level in incoming:
        pressure = 0.75 * pressure + 0.25 * level * level / (1 + 0.3 * material)
        peak = max(peak, pressure)
    return peak


def episode(events, policy, threshold=0.5):
    health = 8
    crossings = work = misses = offload = 0
    for flow, slope, stock, warning, disturbance, hazards in events:
        if health <= 0:
            break
        material = stock
        cost = 0
        if policy == "reactive":
            cross = flow < 0.75
        elif policy == "approximate":
            cross = flow * flow / (1 + 0.3 * stock) < 0.55
            cost = 1
        else:
            active = policy not in ("adaptive", "evolved") or warning < threshold
            if active:
                if policy == "hybrid" and stock < 3:
                    material += 1
                    health -= (
                        1  # Additional construction resources come from forage time.
                    )
                estimate = river_load(flow, slope, material)
                cross = estimate < 0.55
                cost = 3
            else:
                cross = flow < 0.75
        if policy == "external":
            offload += cost
            cost = int(cost > 0)  # Communication uses a local attention tick.
        for i, hazard in enumerate(hazards):
            busy = i < cost
            if policy == "interrupt" and hazard and busy:
                cost = i
                cross = False
                busy = False
            work += busy
            if busy and hazard:
                misses += 1
                health -= 2
            if health <= 0:
                break
        if health <= 0:
            break
        if cross:
            safe = river_load(flow + disturbance, slope, material) < 0.55
            crossings += safe
            health += 2 if safe else -4
        else:
            health -= 1
        health = min(10, health)
    return crossings, work, misses, offload, int(health > 0)


def event_sets(rng, episodes, hazard_rate):
    sets = []
    for _ in range(episodes):
        events = []
        for _ in range(24):
            warning = rng.random()
            flow = rng.uniform(0.2, 1.4)
            slope = rng.uniform(-0.07, 0.07)
            material = int(rng.integers(1, 4))
            hazards = rng.random(3) < hazard_rate * (0.25 + 1.5 * warning)
            events.append(
                (flow, slope, material, warning, rng.normal(0, 0.06), hazards)
            )
        sets.append(events)
    return sets


def river(seed):
    rng = np.random.default_rng(seed)
    training = event_sets(rng, 12, 0.12)

    def fitness(threshold):
        outcomes = [episode(e, "evolved", threshold) for e in training]
        return sum(x[0] + 4 * x[4] for x in outcomes)

    threshold = 0.5
    score = fitness(threshold)
    for _ in range(8):
        proposal = float(np.clip(threshold + rng.normal(0, 0.2), 0, 1))
        candidate = fitness(proposal)
        if candidate > score:
            threshold, score = proposal, candidate
    result = {"evolved_threshold": threshold}
    for environment, hazard_rate in (("quiet", 0.01), ("hazardous", 0.20)):
        tests = event_sets(rng, 24, hazard_rate)
        for policy in (
            "reactive",
            "approximate",
            "unrestricted",
            "interrupt",
            "adaptive",
            "external",
            "hybrid",
            "evolved",
        ):
            outcomes = np.array(
                [
                    episode(e, policy, threshold if policy == "evolved" else 0.5)
                    for e in tests
                ]
            )
            for j, metric in enumerate(
                ("crossings", "work", "misses", "external_ticks", "survival")
            ):
                result[f"{environment}_{policy}_{metric}"] = float(
                    outcomes[:, j].mean()
                )
    return result
