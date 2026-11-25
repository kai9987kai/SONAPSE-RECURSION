"""Embed trained parameters and measured benchmark summaries for offline use."""

import base64
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    from sonapse_recursion.emulation import train
    from sonapse_recursion.targets import Circuit, Physical
    from sonapse_recursion.experiments import evolution

    models = {}
    circuit = Circuit.random(3, np.random.default_rng(83))
    for name, target in [("circuit", circuit), ("pendulum", Physical("pendulum"))]:
        host, _, _ = train(target, 84)
        models[name] = dict(
            project=host.projection.tolist(),
            readout=host.readout.tolist(),
            size=target.size,
        )
        if name == "circuit":
            models[name].update(
                weights=circuit.weights.tolist(), ports=circuit.ports.tolist()
            )
            outer, _, _ = train(host, 85)
            models["nested"] = dict(
                project=outer.projection.tolist(),
                readout=outer.readout.tolist(),
                size=3,
            )
    models["evolution"] = []
    evolution(0, models["evolution"])
    studies = json.loads((ROOT / "results/report.json").read_text())["studies"]
    report = {
        n: {
            "mean": r["mean"],
            "intervals": r["bootstrap_95"],
            "seeds": len(r["trials"]),
        }
        for n, r in studies.items()
    }
    page = (ROOT / "web/laboratory.html.in").read_text(encoding="utf8")
    clips = {
        path.stem: base64.b64encode(path.read_bytes()).decode("ascii")
        for path in sorted((ROOT / "web/stimuli").glob("*.webm"))
    }
    if set(clips) != {"orbit", "ripple", "crossing"}:
        raise ValueError("Missing sensory example clip")
    sensory = (ROOT / "web/sensory.js").read_text(encoding="utf8")
    page = page.replace(
        "<!--SENSORY_PANEL-->",
        (ROOT / "web/sensory.html.in").read_text(encoding="utf8"),
    )
    page = page.replace(
        "/*SENSORY_CODE*/", sensory.replace("SENSORY_STIMULI_DATA", json.dumps(clips))
    )
    (ROOT / "web/index.html").write_text(
        page.replace("/*MODEL_DATA*/null", json.dumps(models)).replace(
            "/*REPORT_DATA*/null", json.dumps(report)
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    build()
