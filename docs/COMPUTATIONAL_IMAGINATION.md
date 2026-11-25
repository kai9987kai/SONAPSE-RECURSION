# Computational imagination

Each physical reference implements a discrete update. Pendulum and spring targets use a semi-implicit velocity update at time step 0.12. The collision adds boundary projection and velocity restitution. The electrical example is a cascaded RC filter. The flow example exchanges a linear flux between two lumped compartments; it has no spatial mesh or turbulence.

Train and test use independent drive generators. Sixteen 64-step paths from varied initial conditions produce the training pairs. A 256-step test begins from zero and includes an unexpected eight-step drive pulse. Experiments record one-step, free-running and final-quarter errors. Runs compare a fitted linear transition and a zero-initial-state hold with the nonlinear host. Scheduled sampling mixes teacher contexts with model contexts during a second fit.

The feature-expanded host has a finite activation range of ±8. Saturation fractions expose unstable rollouts. Outputs are never corrected to the reference during an autonomous run. Physical fidelity is relative to these particular discrete equations.

Additional reference targets include a nonlinear oscillator, damped motor controller, gated memory, associative pattern recall, RLC circuit, bouncing projectile, coupled linkage, lumped dam load, shift register and seven-cell automaton. Each runs through the same learned, linear, scheduled and autonomous evaluation paths.
