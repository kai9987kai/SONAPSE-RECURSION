# SONAPSE-RECURSION

A CPU research laboratory for testing whether a learned neural transition can maintain another system's state, and whether a host can use the resulting computation. The first release measures autonomous emulation, fixed-resource grafts, nested state, interruptions and simulated communication.

**Open [the standalone laboratory](web/index.html)** directly from disk. Select a workspace, press Run or Single step, and use Reset / replay to repeat its input. Open Backend verification results for the measured seed distributions. The offline file includes trained parameters and benchmark summaries.

## Run the research backend

From this directory in PowerShell, with Python 3.10 or later installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m sonapse_recursion --seeds 8 --workers 3
.\.venv\Scripts\python -m unittest discover -s tests -v
```

To repeat one experiment independently use `python -m sonapse_recursion --study recurrent --first-seed 100 --seeds 8 --out results/replication`. Worker count changes scheduling only. The runner commits each trial to SQLite and atomically exports JSON. See [reproduction details](docs/REPRODUCIBILITY.md).

## Repository guide

| Location | Contents |
|---|---|
| sonapse_recursion/targets.py, substrate.py | Independent reference equations and configurable neural substrate |
| sonapse_recursion/host.py, emulation.py | Learned transitions, independent rollout and graph-conditioned deployment |
| sonapse_recursion/resources.py, storage.py | Bounded state, hierarchy, interruption and portable checkpoints |
| sonapse_recursion/experiments.py, augments.py, hierarchy.py, design.py | Comparative studies and synthetic interventions |
| sonapse_recursion/runner.py | Seed isolation, worker control, resumable trials and export |
| web/ | Self-contained dashboard and source template |
| scripts/ | Build the offline page from trained models and measured results |
| tests/ | Backend regressions and optional real-browser checks |
| results/report.json | Individual trials, process CPU times, means and seed-bootstrap intervals |
| docs/ | Experimental protocols, limitations, background and development plan |

Read the [benchmark interpretation](docs/BENCHMARKS.md), [hypotheses](docs/RESEARCH_HYPOTHESES.md) and [evidence limits](docs/SCIENTIFIC_LIMITATIONS.md) before interpreting a displayed metric. All reported tasks are synthetic.

## Local research dashboard

Run `python -m sonapse_recursion.dashboard` from the project root and open the loopback address it prints. In Backend verification results select a study, enter a first seed and seed count, and press Run benchmark. The local worker records results under results/interactive. Opening the HTML directly provides the offline replay and recorded results.

Run `python -m sonapse_recursion.replicate` for the independent-seed follow-up or `python -m sonapse_recursion --profile large-connectome --out results/sparse` for the 4,096-node event-driven experiment.

See [implementation coverage](docs/CAPABILITY_STATUS.md) for the outstanding requirements in each subsystem.
