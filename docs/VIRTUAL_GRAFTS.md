# Virtual grafts

A graft uses two scalar coordinates from a fixed host bank as persistent memory. At each tick it receives the scalar drive, updates those coordinates with a trained neural transition and exposes the two outputs to a fitted native decoder. The task is recovery of one component of the original target circuit. The decoder sees identical labels, fitting windows and drive sequences across conditions.

A target-specific trained graft, a same-width graft trained on another target with the same data budget, a simple random recurrent control and a disconnected output distinguish specialization, generic memory and direct input decoding. The target trajectory is used for training and scoring only. Matching the alternative-target model controls dimensions and training examples; the simple random control is cheaper.

Bank allocation supports creation, inspection, suspension, resumption and recursive deletion. Model coefficients and feature computation are separately allocated in software. Specialized target pretraining supplies relevant information. Transfer to an unrelated native task, learned graft discovery and a total physical host-wide budget are open.
