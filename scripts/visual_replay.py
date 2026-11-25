"""Independently replay exported browser observations through the Python host."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def advance(previous, pixels):
    image = np.asarray(pixels, dtype=float).reshape(12, 20, 3) / 255
    luminance = image @ [0.2126, 0.7152, 0.0722]
    receptors = np.array(
        [
            luminance[
                (i // 6) * 4 : (i // 6 + 1) * 4,
                (i % 6) * 20 // 6 : (i % 6 + 1) * 20 // 6,
            ].mean()
            for i in range(18)
        ]
    )
    return np.tanh(
        0.4 * previous
        + 1.4 * (receptors - 0.5)
        + 0.48 * np.roll(previous, 1)
        - 0.29 * np.roll(previous, 5)
        + 0.21 * np.roll(previous, 11)
    )


def replay_studio(frames, host):
    """Recompute the 96-region trajectory from the exported precompression samples."""
    checked = 0
    sensory_error = host_error = 0.0
    epoch = None
    for row in sorted(frames, key=lambda r: (r["epoch"], r["tick"])):
        if row["epoch"] != epoch:
            epoch = row["epoch"]
            state = np.zeros(96)
            core = np.zeros(3)
            expected = 0
        if row["tick"] != expected:
            raise ValueError("Noncontiguous studio episode")
        desc = np.asarray(row["desc"], dtype=float)
        observed = np.asarray(row["state"], dtype=float)
        observed_host = np.asarray(row["host"], dtype=float)
        if (
            desc.shape != (288,)
            or observed.shape != (96,)
            or observed_host.shape != (3,)
            or not all(np.isfinite(x).all() for x in (desc, observed, observed_host))
            or (desc < 0).any()
            or (desc > 255).any()
        ):
            raise ValueError("Invalid studio state")
        light = desc.reshape(96, 3) @ [0.213, 0.715, 0.072] / 255
        state = state.astype(np.float64)
        state = np.tanh(
            0.4 * state
            + 1.4 * (light - 0.5)
            + 0.48 * np.roll(state, 1)
            - 0.29 * np.roll(state, 12)
            + 0.21 * np.roll(state, 37)
        ).astype(np.float32)
        core = host.step(core, float(state.mean(dtype=np.float64)))
        sensory_error = max(sensory_error, float(np.max(np.abs(state - observed))))
        host_error = max(host_error, float(np.max(np.abs(core - observed_host))))
        checked += 1
        expected += 1
    return dict(
        frames=len(frames),
        replayed=checked,
        max_sensory_error=sensory_error,
        max_host_error=host_error,
    )


def replay(data):
    from sonapse_recursion.emulation import train
    from sonapse_recursion.targets import Circuit

    if data.get("schema") == "sonapse-studio-v1":
        host, _, _ = train(Circuit.random(3, np.random.default_rng(83)), 84)
        return replay_studio(data["frames"], host)
    if data.get("schema") != "sonapse-visual-v2" or (
        data.get("width"),
        data.get("height"),
    ) != (20, 12):
        raise ValueError("Unknown visual record format")
    frames = data["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 240:
        raise ValueError("Expected 1–240 observations")
    host, _, _ = train(Circuit.random(3, np.random.default_rng(83)), 84)
    previous = None
    max_visual = max_host = 0.0
    verified = 0
    for frame in frames:
        pixels = np.asarray(frame["pixels"])
        observed = np.asarray(frame["state"], dtype=float)
        observed_host = np.asarray(frame["host"], dtype=float)
        if (
            pixels.shape != (720,)
            or pixels.dtype.kind not in "iu"
            or pixels.min() < 0
            or pixels.max() > 255
        ):
            raise ValueError("Invalid quantized frame")
        if (
            observed.shape != (18,)
            or observed_host.shape != (3,)
            or not all(np.isfinite(v).all() for v in (observed, observed_host))
        ):
            raise ValueError("Invalid neural state")
        if frame["tick"] == 0:
            state = np.zeros(18)
            core = np.zeros(3)
        elif previous is None:
            state, core = observed.copy(), observed_host.copy()
            previous = frame
            continue  # First retained frame of a truncated ring buffer seeds replay.
        elif (
            previous["epoch"] != frame["epoch"] or previous["tick"] + 1 != frame["tick"]
        ):
            raise ValueError("Broken observation sequence")
        state = advance(state, pixels)
        core = host.step(core, float(np.mean(state)))
        max_visual = max(max_visual, float(np.max(np.abs(state - observed))))
        max_host = max(max_host, float(np.max(np.abs(core - observed_host))))
        previous = frame
        verified += 1
    return {
        "frames": len(frames),
        "replayed": verified,
        "max_sensory_error": max_visual,
        "max_host_error": max_host,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    result = replay(json.loads(args.record.read_text(encoding="utf8")))
    print(json.dumps(result, indent=2))
    if (
        not result["replayed"]
        or max(result["max_host_error"], result["max_sensory_error"]) > 1e-8
    ):
        raise SystemExit(1)
