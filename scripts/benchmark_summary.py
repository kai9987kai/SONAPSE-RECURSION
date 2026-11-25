"""Generate the research summary from the checked-in trial-level export."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "results/report.json").read_text())
studies = report["studies"]
lines = [
    "# Benchmark results",
    "",
    f"{len(studies)} synthetic studies, seeds 0–7, {sum(len(x['trials']) for x in studies.values())} trials. Each row below compares two conditions on the same seed. The interval resamples the paired differences with replacement 2,000 times. With eight independent seeds, these are exploratory uncertainty estimates.",
    "",
    "| Study and outcome | First condition | Comparison | First minus comparison, mean [95% interval] |",
    "|---|---:|---:|---:|",
]
pairs = {
    "single": ("autonomous_rmse", "linear_rmse"),
    "recurrent": ("autonomous_rmse", "linear_rmse"),
    "pendulum": ("autonomous_rmse", "linear_rmse"),
    "lif": ("autonomous_rmse", "linear_rmse"),
    "collision": ("autonomous_rmse", "linear_rmse"),
    "plastic": ("autonomous_rmse", "linear_rmse"),
    "structural": ("unseen_rmse", "shuffled_edges_rmse"),
    "graft": ("trained_rmse", "other_target_rmse"),
    "hierarchy": ("distilled_child_rmse", "flat_child_rmse"),
    "interference": ("shared_rmse", "isolated_rmse"),
    "channel": ("predictive_rmse", "hold_rmse"),
    "closed_loop": ("predict_rms", "hold_rms"),
    "evolution": ("evolved_test_rmse", "independent_search_rmse"),
    "adaptation": ("reciprocal_rmse", "decoder_only_rmse"),
    "external": ("external_delayed_rmse", "native_rmse"),
    "river": ("hazardous_unrestricted_survival", "hazardous_reactive_survival"),
}
for name, (a, b) in pairs.items():
    r = studies[name]
    diffs = np.array([t["metrics"][a] - t["metrics"][b] for t in r["trials"]])
    samples = np.random.default_rng(491).choice(diffs, (2000, len(diffs)))
    lo, hi = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    lines.append(
        f"| {name} | {a.replace('_', ' ')} · {r['mean'][a]:.5g} | {b.replace('_', ' ')} · {r['mean'][b]:.5g} | {diffs.mean():.5g} [{lo:.5g}, {hi:.5g}] |"
    )
lines.extend(
    [
        "",
        "## Interpretation",
        "",
        "Smooth rate circuits were substantially easier to roll out than spiking, plastic or colliding targets. Spring, RC and flow have simple linear dynamics, making the fitted linear control particularly strong.",
        "",
        "The graft benefited a task aligned with its own pretraining. The nested two-clock learner did not improve mean distal-child error over a flat learner. In the river experiment, unrestricted planning helped under quiet conditions and incurred a survival cost under frequent hazards. The open-loop predictor improved signal reconstruction, while applying extrapolation inside a feedback loop increased state error.",
        "",
        "## Coverage",
        "",
        "The full export includes seed-level results and timing for every study. Native task accuracy, RMSE, probabilities, work ticks and architecture parameters use different units. Compare values only within an outcome.",
        "",
        ", ".join(studies),
        "",
        "## Replication and imported graph",
        "",
        "The bounded replication command selects the lowest and highest mean autonomous-to-linear RMSE ratios, freezes this selection and runs seeds 100–107. Read [the protocol](../results/replication/protocol.json) and [the observed comparison](../results/replication/comparison.json).",
        "",
        "A separate imported-graph probe used an alphabetical induced subgraph of 25 cells and 86 nonzero directed chemical edges. Both models received a shared scalar drive. Learned versus linear RMSE was 0.00791 versus 0.02506 on one rollout. These are synthetic dynamics with normalized anatomical counts. [Import provenance and exact nodes](../results/imported.json).",
        "",
        "[Complete machine-readable results](../results/report.json).",
    ]
)
(ROOT / "docs/BENCHMARKS.md").write_text("\n".join(lines) + "\n", encoding="utf8")
