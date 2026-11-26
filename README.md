# SONAPSE-RECURSION

A CPU research laboratory for testing whether a learned neural transition can maintain another system's state, and whether a host can use the resulting computation. The first release measures autonomous emulation, fixed-resource grafts, nested state, interruptions and simulated communication.

**Open [the standalone laboratory](web/index.html)** directly from disk. Select a workspace, press Run or Single step, and use Reset / replay to repeat its input. Open Backend verification results for the measured seed distributions. The offline file includes trained parameters and benchmark summaries.

## Multistream imagination

Open **Multistream imagination** and add several examples, files, a webcam or a direct video URL. The studio generates predictions up to 1920 × 1080 and retains the full visual observation archive in this browser. Pick a starting episode and a rollout horizon to imagine autonomously. [Studio controls, memory and replay](docs/MULTISTREAM.md).

The large-connectome profile now runs the original 4,096-node study and a 65,536-node, 131,072-edge study. Select only the larger study with `python -m sonapse_recursion --study sparse_65k --seeds 8 --out results/large`.

The [32-seed scale run](results/scale/report.json) covers every registered study. See [reproduction settings](docs/REPRODUCIBILITY.md) for the command and seed ranges.

## Live visual input and imagination

Open **Sensory imagination** in the dashboard to feed one of three included videos, a local file, or a webcam. Watch the recurrent graph and the generated visual continuation side by side. Choose a remembered episode to imagine after capture stops. Memory stays local and can be saved or exported. See [sensory controls and model details](docs/SENSORY_IMAGINATION.md) and the [41-section master prompt audit](docs/MASTER_PROMPT_AUDIT.md).

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

## Dreaming

Open Dreaming in the multistream studio and choose Start dreaming. The studio releases live inputs. A seeded scene composer associates fragments across available episodes, invents landscapes and carries a changing object across scenes. An empty archive works too. The generated image drives the visual connectome, whose output influences subsequent rendering. Pause, single-step and wake controls allow inspection. Save the full-resolution PNG, a six-scene strip, or a JSON journal of the last 96 scene plans and their memory sources. The journal and strip stay in session until the next dream or archive clear.

This is an artistic and computational model of dreamlike recombination. Its scenes and transition rules are designed by hand. Human dream reports provided inspiration for fragmentary recall. [Fosse et al., 2003](https://pubmed.ncbi.nlm.nih.gov/12590838/).
