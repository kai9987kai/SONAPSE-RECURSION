"""Portable, validated JSON for host parameters, recurrent state, and virtual time."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from .host import Host
from .targets import Circuit


def save(path, data):
    path = Path(path)
    payload = json.dumps(data, allow_nan=False)
    with NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf8"
    ) as stream:
        stream.write(payload)
    os.replace(stream.name, path)


def save_host(path, host, state, tick):
    save(
        path,
        dict(
            format="host-1",
            size=host.size,
            project=host.projection.tolist(),
            readout=host.readout.tolist(),
            state=np.asarray(state).tolist(),
            tick=tick,
        ),
    )


def load_host(path):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    n, tick = data["size"], data["tick"]
    if (
        data["format"] != "host-1"
        or type(n) is not int
        or n < 1
        or type(tick) is not int
        or tick < 0
    ):
        raise ValueError("Invalid checkpoint header")
    project, readout, state = (
        np.asarray(data[k], dtype=float) for k in ("project", "readout", "state")
    )
    if (
        project.ndim != 2
        or project.shape[0] != n + 2
        or readout.shape != (n + 2 + project.shape[1], n)
        or state.shape != (n,)
        or not all(np.isfinite(x).all() for x in (project, readout, state))
    ):
        raise ValueError("Invalid checkpoint arrays")
    host = Host(n, 0, np.random.default_rng(0))
    host.projection, host.readout = project, readout
    return host, state, tick


def save_circuit(path, graph):
    save(
        path,
        dict(
            format="circuit-1",
            weights=graph.weights.tolist(),
            ports=graph.ports.tolist(),
        ),
    )


def load_circuit(path):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data["format"] != "circuit-1":
        raise ValueError("Unknown architecture format")
    ports = np.asarray(data["ports"])
    weights = np.asarray(data["weights"])
    if ports.shape == (0,) and weights.shape == (0,):
        weights = weights.reshape(0, 0)
    return Circuit(weights, ports)
