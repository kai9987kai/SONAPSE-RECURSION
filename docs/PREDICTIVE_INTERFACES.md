# Predictive interface

A synthetic sensor sends scalar samples through a channel with random 2–6 tick delay and independent 12% loss. The receiver rejects packets older than its newest timestamp. Prediction uses a two-lag linear model fitted to the first half of the signal. Scoring uses the second half. Extrapolation requires consecutive transmitted timestamps, has an eight-tick limit, and clips to a specified signal range. Otherwise the receiver holds the latest value.

The control holds the latest value on every tick using precisely the same arrivals. The browser has a separate slope extrapolation demonstration with identical causal delivery rules. A periodic source favors prediction. Abrupt regime changes, calibration of prediction confidence and closed-loop instability require additional studies.

The closed-loop experiment controls a noisy scalar plant with a disturbance burst. It compares hold, bounded slope extrapolation and confidence weighting based on the error of past forecasts. Delays and disturbances match across policies. State RMS and peak absolute excursion expose cases where a useful signal predictor destabilizes control.
