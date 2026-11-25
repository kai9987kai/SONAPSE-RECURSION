# Hypotheses and null comparisons

The experimental unit is an independently seeded synthetic trial. Each comparison uses paired stimuli unless the task explicitly tests transfer. Reduced error or increased survival is the prespecified direction of interest. These exploratory comparisons do not constitute a confirmatory study.

| Hypothesis | Operational test | Null or rival explanation |
|---|---|---|
| H1 Neural dynamics | Autonomous single, recurrent, LIF and plastic rollouts | Fitted linear dynamics achieve equal or lower error |
| H2 Structural transfer | Train a shared local rule on five-node graphs, evaluate ten-node graphs | Shuffling graph rows preserves accuracy |
| H3 Graft benefit | Train a native decoder with a virtual graft port | Same-sized alternative-target training gives equivalent error |
| H4 Resource competition | Compare uninterrupted work with native attention | Work causes no survival change |
| H5 Dual imagination | Inject identical noise through different state gates | Gate strength has no effect |
| H6 Evolution | Select inherited edge mutations against independent trials with equal search count | Improvement is explained by candidate search alone |
| H7 Survival paradox | Bound attention during a concurrent workload | Hazard survival is unchanged |
| H8 Governor | Pause work upon threat or protect the attention lane | Recovery matches uninterrupted workload |
| H9 Co-adaptation | Learn both directions of a scalar channel | Decoder-only learning performs equally well |
| H10 Prediction | Predict delayed packets using a causal fitted predictor | Holding the last observation gives equal error |
| H11 Recursion | Distil an inner host and score a separate native state and child clock | A direct flat learner matches the distilled learner |
| H12 External resources | Compare four native cells with a delayed sixteen-cell external reservoir | Native-only or colocated computation is as effective |

A formal generalisation claim requires fresh architecture families, paired effect intervals, objective selection before observing outcomes and replication by an independent implementation.
