"""Run bounded studies. SQLite commits at trial boundaries for resumable execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
from contextlib import closing
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import experiments as ex
from . import augments
from .design import design
from .hierarchy import hierarchy
from . import discovery
from .emulation import structural, study
from .domains import Domain
from .sparse import sparse_study
from .river import river
from .governor import governance
from .binding import binding

NAMES = (
    "logic",
    "single",
    "recurrent",
    "plastic",
    "pendulum",
    "spring",
    "collision",
    "rc",
    "flow",
    "structural",
    "graft",
    "nested",
    "interference",
    "survival",
    "channel",
    "adaptation",
    "evolution",
    "lif",
    "external",
    "interruption",
    "machine",
    "design",
    "hierarchy",
    "curriculum",
    "pareto",
    "closed_loop",
    "sparse",
    "river",
    "governance",
    "binding",
    *Domain.sizes,
)


def execute(job):
    name, seed = job
    start = time.process_time()
    if name in ("single", "recurrent"):
        values = study("circuit", seed, 1 if name == "single" else 5)
    elif name in (
        "plastic",
        "pendulum",
        "spring",
        "collision",
        "rc",
        "flow",
        "lif",
        *Domain.sizes,
    ):
        values = study(name, seed)
    elif name == "structural":
        values = structural(seed)
    elif name == "design":
        values = design(seed)
    elif name == "sparse":
        values = sparse_study(seed)
    elif name == "river":
        values = river(seed)
    elif name == "governance":
        values = governance(seed)
    elif name == "binding":
        values = binding(seed)
    elif name == "hierarchy":
        values = hierarchy(seed)
    elif name in ("curriculum", "pareto", "closed_loop"):
        values = getattr(discovery, name)(seed)
    elif name in ("external", "interruption", "machine"):
        values = getattr(augments, name)(seed)
    else:
        values = getattr(ex, name)(seed)
    return name, seed, values, time.process_time() - start


def fingerprint():
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.read_bytes())
    digest.update((np.__version__ + platform.python_version()).encode())
    return digest.hexdigest()


def run(names, seeds, folder, workers):
    folder.mkdir(parents=True, exist_ok=True)
    version = fingerprint()
    with closing(sqlite3.connect(folder / "trials.sqlite", timeout=60)) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS trials (version,name,seed,metrics,cpu,PRIMARY KEY(version,name,seed))"
        )
        rows = {
            (row[0], row[1]): row[2:]
            for row in db.execute(
                "SELECT name,seed,metrics,cpu FROM trials WHERE version=?", (version,)
            )
        }
        jobs = [(n, s) for n in names for s in seeds if (n, s) not in rows]
        pool = ProcessPoolExecutor(workers) if workers > 1 and jobs else None
        try:
            for n, s, v, cpu in pool.map(execute, jobs) if pool else map(execute, jobs):
                data = json.dumps(v, allow_nan=False)
                db.execute(
                    "INSERT OR IGNORE INTO trials VALUES(?,?,?,?,?)",
                    (version, n, s, data, cpu),
                )
                db.commit()
                rows[n, s] = data, cpu
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
    report = dict(
        source_sha256=version,
        python=platform.python_version(),
        numpy=np.__version__,
        studies={},
    )
    for n in names:
        trials = [
            dict(seed=s, metrics=json.loads(rows[n, s][0]), cpu_seconds=rows[n, s][1])
            for s in seeds
        ]
        means = {
            k: float(np.mean([t["metrics"][k] for t in trials]))
            for k in trials[0]["metrics"]
        }
        # Same resampled seeds across metrics preserve pairing.
        draws = np.random.default_rng(491).integers(
            len(trials), size=(2000, len(trials))
        )
        ci = (
            {
                k: np.quantile(
                    np.array([t["metrics"][k] for t in trials])[draws].mean(axis=1),
                    [0.025, 0.975],
                ).tolist()
                for k in means
            }
            if len(trials) > 1
            else {}
        )
        report["studies"][n] = dict(trials=trials, mean=means, bootstrap_95=ci)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=folder, encoding="utf8"
    ) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.replace(stream.name, folder / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profiles = {
        "headless": (NAMES, 5, 1),
        "research": (NAMES, 8, 2),
        "educational": (("single", "pendulum", "graft"), 1, 1),
        "evolution": (("evolution", "pareto", "design"), 8, 3),
        "large-connectome": (("sparse",), 5, 1),
        "hardware-limited": (("single", "graft", "channel"), 2, 1),
    }
    parser.add_argument("--profile", choices=profiles, default="headless")
    parser.add_argument("--study", choices=("all", *NAMES))
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    names, default_seeds, default_workers = profiles[args.profile]
    if args.seeds is None:
        args.seeds = default_seeds
    if args.workers is None:
        args.workers = default_workers
    if args.seeds < 1 or args.first_seed < 0 or not 1 <= args.workers <= 8:
        parser.error("Use positive seed count, nonnegative start, and 1–8 workers")
    if args.study is not None:
        names = NAMES if args.study == "all" else (args.study,)
    run(
        names,
        range(args.first_seed, args.first_seed + args.seeds),
        args.out,
        args.workers,
    )
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
