"""Select a failure and a success from a frozen pilot, then test on fresh seeds."""

import argparse
import hashlib
import json
from pathlib import Path
from .runner import run, fingerprint
from .storage import save


def replicate(pilot, folder, first, count):
    folder.mkdir(parents=True, exist_ok=True)
    raw = pilot.read_bytes()
    report = json.loads(raw)
    if report["source_sha256"] != fingerprint():
        raise ValueError("Pilot method differs from installed source")
    if count < 1 or count > 100 or first < 0:
        raise ValueError("Replication budget must be 1–100 nonnegative seeds per study")
    if (folder / "report.json").resolve() == pilot.resolve():
        raise ValueError("Choose a separate replication directory")
    seeds = range(first, first + count)
    studied = {t["seed"] for s in report["studies"].values() for t in s["trials"]}
    if studied.intersection(seeds):
        raise ValueError("Replication overlaps pilot seeds")
    ratios = {
        k: s["mean"]["autonomous_rmse"] / max(1e-12, s["mean"]["linear_rmse"])
        for k, s in report["studies"].items()
        if "autonomous_rmse" in s["mean"]
    }
    if len(ratios) < 2:
        raise ValueError("Need two eligible neural versus linear pilot studies")
    ranked = sorted(ratios, key=ratios.get)
    names = [ranked[0], ranked[-1]]
    plan = dict(
        pilot_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=fingerprint(),
        rule="smallest and largest pilot ratios of mean autonomous to mean linear RMSE",
        pilot_ratios=ratios,
        selected=names,
        first_seed=first,
        seeds_per_study=count,
        outcome="direction of difference in mean RMSE on independent seeds",
    )
    save(folder / "protocol.json", plan)
    followup = run(names, seeds, folder, 1)
    conclusions = {}
    for n in names:
        m = followup["studies"][n]["mean"]
        ratio = m["autonomous_rmse"] / max(1e-12, m["linear_rmse"])
        conclusions[n] = dict(
            pilot_ratio=ratios[n],
            replication_ratio=ratio,
            direction_repeated=(ratio < 1) == (ratios[n] < 1),
        )
    save(folder / "comparison.json", conclusions)
    return conclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/report.json"))
    parser.add_argument("--out", type=Path, default=Path("results/replication"))
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            replicate(args.pilot, args.out, args.first_seed, args.seeds), indent=2
        )
    )


if __name__ == "__main__":
    main()
