# Benchmark results

40 synthetic studies, seeds 0–7, 320 trials. Each row below compares two conditions on the same seed. The interval resamples the paired differences with replacement 2,000 times. With eight independent seeds, these are exploratory uncertainty estimates.

| Study and outcome | First condition | Comparison | First minus comparison, mean [95% interval] |
|---|---:|---:|---:|
| single | autonomous rmse · 0.00028947 | linear rmse · 0.028811 | -0.028521 [-0.048836, -0.011047] |
| recurrent | autonomous rmse · 0.0028063 | linear rmse · 0.016171 | -0.013365 [-0.017173, -0.0099189] |
| pendulum | autonomous rmse · 0.055106 | linear rmse · 0.21926 | -0.16416 [-0.32981, -0.037971] |
| lif | autonomous rmse · 7.8203 | linear rmse · 0.29371 | 7.5266 [7.4259, 7.605] |
| collision | autonomous rmse · 0.88895 | linear rmse · 0.23078 | 0.65817 [0.024331, 1.8853] |
| plastic | autonomous rmse · 1.4595 | linear rmse · 0.2134 | 1.2461 [0.23809, 2.6329] |
| structural | unseen rmse · 0.0019565 | shuffled edges rmse · 0.088486 | -0.08653 [-0.09613, -0.076688] |
| graft | trained rmse · 0.00037182 | other target rmse · 0.020204 | -0.019832 [-0.030761, -0.010639] |
| hierarchy | distilled child rmse · 0.0018486 | flat child rmse · 0.0018219 | 2.6683e-05 [-3.0457e-05, 0.00010807] |
| interference | shared rmse · 0.10535 | isolated rmse · 0.00075814 | 0.1046 [0.098885, 0.11015] |
| channel | predictive rmse · 0.26485 | hold rmse · 0.31406 | -0.049212 [-0.056348, -0.041504] |
| closed_loop | predict rms · 0.56323 | hold rms · 0.29968 | 0.26354 [0.2244, 0.30406] |
| evolution | evolved test rmse · 0.11612 | independent search rmse · 0.12958 | -0.01346 [-0.026939, -0.0040437] |
| adaptation | reciprocal rmse · 0.043779 | decoder only rmse · 0.17071 | -0.12693 [-0.18882, -0.064307] |
| external | external delayed rmse · 0.071938 | native rmse · 0.32246 | -0.25053 [-0.26617, -0.23389] |
| river | hazardous unrestricted survival · 0.067708 | hazardous reactive survival · 0.58333 | -0.51562 [-0.57292, -0.45833] |

## Interpretation

Smooth rate circuits were substantially easier to roll out than spiking, plastic or colliding targets. Spring, RC and flow have simple linear dynamics, making the fitted linear control particularly strong.

The graft benefited a task aligned with its own pretraining. The nested two-clock learner did not improve mean distal-child error over a flat learner. In the river experiment, unrestricted planning helped under quiet conditions and incurred a survival cost under frequent hazards. The open-loop predictor improved signal reconstruction, while applying extrapolation inside a feedback loop increased state error.

## Coverage

The full export includes seed-level results and timing for every study. Native task accuracy, RMSE, probabilities, work ticks and architecture parameters use different units. Compare values only within an outcome.

logic, single, recurrent, plastic, pendulum, spring, collision, rc, flow, structural, graft, nested, interference, survival, channel, adaptation, evolution, lif, external, interruption, machine, design, hierarchy, curriculum, pareto, closed_loop, sparse, river, governance, binding, oscillator, motor, memory, associative, rlc, ballistic, linkage, dam_load, register, automaton

## Replication and imported graph

The bounded replication command selects the lowest and highest mean autonomous-to-linear RMSE ratios, freezes this selection and runs seeds 100–107. Read [the protocol](../results/replication/protocol.json) and [the observed comparison](../results/replication/comparison.json).

A separate imported-graph probe used an alphabetical induced subgraph of 25 cells and 86 nonzero directed chemical edges. Both models received a shared scalar drive. Learned versus linear RMSE was 0.00791 versus 0.02506 on one rollout. These are synthetic dynamics with normalized anatomical counts. [Import provenance and exact nodes](../results/imported.json).

[Complete machine-readable results](../results/report.json).
