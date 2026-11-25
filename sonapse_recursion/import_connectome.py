"""Import OpenWorm-format anatomical edges and run an explicitly synthetic rate probe."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
import numpy as np
from .targets import Circuit, reference
from .emulation import signals, train, rmse
from .storage import save


def import_csv(path, limit):
    if not 1 <= limit <= 256:
        raise ValueError("Node limit must be 1–256")
    raw = Path(path).read_bytes()
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not {"Source", "Target", "Weight", "Type"}.issubset(rows.fieldnames or ()):
        raise ValueError("Missing edge columns")
    edges = []
    for row in rows:
        source, target = row["Source"].strip(), row["Target"].strip()
        weight = float(row["Weight"])
        if not source or not target or not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid anatomical edge")
        if row["Type"].strip().lower() == "chemical":
            edges.append((source, target, weight))
    nodes = sorted({node for a, b, _ in edges for node in (a, b)})[:limit]
    if not nodes:
        raise ValueError("No chemical edges")
    ids = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    for source, target, weight in edges:
        if source in ids and target in ids:
            matrix[ids[target], ids[source]] += weight
    matrix /= max(1, np.linalg.norm(matrix) / 0.85)
    return Circuit(matrix, np.ones(len(nodes))), nodes, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("results/imported.json"))
    args = parser.parse_args()
    graph, names, checksum = import_csv(args.csv, args.limit)
    u = signals(np.random.default_rng(91), 256)
    host, linear, _ = train(graph, 90)
    truth = reference(graph, u)[1:]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(
        args.out,
        dict(
            source=args.source,
            sha256=checksum,
            selection="alphabetical induced chemical subgraph",
            nodes=names,
            edges=int(np.count_nonzero(graph.weights)),
            interpretation="anatomical counts, normalized synthetic rate dynamics",
            seed_train=90,
            seed_test=91,
            steps=256,
            autonomous_rmse=rmse(host.rollout(u, np.zeros(graph.size)), truth),
            linear_rmse=rmse(linear.rollout(u, np.zeros(graph.size)), truth),
        ),
    )


if __name__ == "__main__":
    main()
