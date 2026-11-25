# Architecture and provenance

The inspected workspace contained no project files or Git history. The development sequence was to isolate the scoring reference, train and verify a recurrent transition, allocate its state inside a fixed bank, test interventions with shared stimuli, and then expose the verified experiments through a headless runner and an offline dashboard.

Reference targets own their equations. Training obtains example transitions from a target. Deployment accepts only an initial state, model parameters and a sequence of inputs. The evaluator separately computes the target trace. A regression test disables the reference step during host execution. A second architecture consumes an explicit graph and uses conventional weighted message aggregation plus a learned local update.

The configurable substrate supports rate, LIF, AdEx-inspired, Izhikevich, compartment, two-soma, dendritic and hybrid units. It implements signed delayed transmission, optional modulated Hebbian updates and activity accounting. These are simplified models. The autonomous accuracy results use the separately trained rate host. A state-bank lease reserves represented state coordinates. Neural projections, readout coefficients and temporary activations consume additional memory.

The inspected [Sonapse-LAB graft bus](https://github.com/kai9987kai/Sonapse-LAB/blob/main/synapse/graft_bus.py) and [NEUROGRAFT package](https://github.com/kai9987kai/NEUROGRAFT/tree/main/neurograft) cover related interface and network research. The local Sonapse-LAB package requires Torch, NumPy and SciPy. This implementation uses original code, uses NumPy alone, and keeps its scorer independently inspectable. Existing sibling repositories and their running services were left alone.

Background on neural dynamics and readouts comes from [Maass, Natschläger and Markram, 2002](https://doi.org/10.1162/089976602760407955). Mixing generated and teacher contexts during training follows the experimental distinction described by [Bengio et al., 2015](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html). Neither paper validates this implementation.

The separate anatomical import probe reads the OpenWorm hermaphrodite edge list, records a SHA-256 fingerprint, keeps chemical edges, and normalizes the induced adjacency. All physiology for this probe is synthetic. The public data file is retrieved separately. [Pinned OpenWorm source](https://github.com/openworm/CElegansNeuroML/blob/b36380a36d2a6dda0f03c946c433524b25ea2268/herm_full_edgelist.csv).

## Multistream visual path

The studio composites independent browser media decoders and optionally records separate source views in the same atomic IndexedDB batch. Its 96-region recurrent visual encoder supplies the existing trained host with a scalar drive. Every archived descriptor participates in spatial context retrieval, while up to three observed successor pairs supply fine-scale motion to the full-resolution output. Generated rollout frames never enter the observation store. Python replay checks saved descriptors, state transitions and host outputs. [Operational details](MULTISTREAM.md).

The sparse simulation engine schedules unique event times with a heap. Input coalescing, deterministic node order and refractory rules remain part of the reference dynamics. The scale profile and chunked paired-bootstrap estimator use the same experiment implementations as the reference suite. [Scale-run settings](REPRODUCIBILITY.md).
