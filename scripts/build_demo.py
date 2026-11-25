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

    def summarize(path):
        studies = json.loads(path.read_text(encoding="utf8"))["studies"]
        return {
            name: dict(
                mean=row["mean"],
                intervals=row["bootstrap_95"],
                seeds=len(row["trials"]),
            )
            for name, row in studies.items()
        }

    report = summarize(ROOT / "results/report.json")
    scaled = summarize(ROOT / "results/scale/report.json")
    page = (ROOT / "web/laboratory.html.in").read_text(encoding="utf8")
    clips = {
        path.stem: base64.b64encode(path.read_bytes()).decode("ascii")
        for path in sorted((ROOT / "web/stimuli").glob("*.webm"))
    }
    if not {
        "orbit",
        "ripple",
        "crossing",
        "julia",
        "mandelbrot",
        "carpet",
        "interference",
        "ribbons",
    } <= set(clips):
        raise ValueError("Missing sensory example clip")
    sensory = (ROOT / "web/sensory.js").read_text(encoding="utf8")
    page = page.replace(
        "<!--SENSORY_PANEL-->",
        (ROOT / "web/studio.html.in").read_text(encoding="utf8")
        + (ROOT / "web/sensory.html.in").read_text(encoding="utf8"),
    )
    page = page.replace(
        "/*SENSORY_CODE*/",
        "const STUDIO_CLIPS="
        + json.dumps(clips)
        + ";\n"
        + sensory.replace("SENSORY_STIMULI_DATA", "STUDIO_CLIPS")
        + (ROOT / "web/studio.js").read_text(encoding="utf8"),
    )
    (ROOT / "web/index.html").write_text(
        page.replace("/*MODEL_DATA*/null", json.dumps(models))
        .replace("/*REPORT_DATA*/null", json.dumps(report))
        .replace("/*SCALE_DATA*/null", json.dumps(scaled)),
        encoding="utf8",
    )


if __name__ == "__main__":
    build()
