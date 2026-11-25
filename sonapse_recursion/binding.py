"""Nested program binding probe. Program selection uses two synthetic opcode bits."""

import numpy as np
from .host import ridge


class BindingKernel:
    """An outer neural state hosts a child's retained opcode and one-bit accumulator."""

    def __init__(self, rng):
        self.projection = rng.normal(size=(7, 128))
        self.coefficients = None

    def features(self, rows):
        raw = np.column_stack((rows, np.ones(len(rows))))
        return np.column_stack((raw, np.tanh(raw @ self.projection)))

    def learn(self, contexts, labels):
        self.coefficients = ridge(self.features(contexts), labels)

    def update(self, contexts):
        return (self.features(contexts) @ self.coefficients > 0.5).astype(int)


def interpreter(state, command):
    p, q, bit = state
    load, a, b = command
    if load:
        return [a, b, 0]
    values = (bit ^ a, bit | a, bit & a, 1 - bit)
    return [p, q, values[2 * p + q]]


def binding(seed):
    rng = np.random.default_rng(seed)
    # The inner teacher learns the reset, binding, and execution rule from examples.
    rows = rng.integers(0, 2, (320, 6))
    labels = np.array([interpreter(row[:3], row[3:]) for row in rows])
    teacher = BindingKernel(rng)
    teacher.learn(rows, labels)
    examples = rng.integers(0, 2, (320, 6))
    projection_seed = seed + 401
    outer = BindingKernel(np.random.default_rng(projection_seed))
    flat = BindingKernel(np.random.default_rng(projection_seed))
    outer.learn(examples, teacher.update(examples))
    flat.learn(examples, np.array([interpreter(row[:3], row[3:]) for row in examples]))
    inputs = rng.integers(0, 2, (256, 3))
    inputs[:, 0] = 0
    inputs[::19, 0] = 1
    expected = np.zeros(3, dtype=int)
    states = np.zeros((3, 3), dtype=int)
    hits = np.zeros((3, 3))
    for tick, command in enumerate(inputs):
        expected = np.array(interpreter(expected, command))
        for i, kernel in enumerate((outer, flat, outer)):
            if i == 2 and tick % 19 == 1:
                states[i, :2] = 1 - states[i, :2]
            states[i] = kernel.update([np.r_[states[i], command]])[0]
            hits[i] += states[i] == expected
    return dict(
        distilled_output_accuracy=hits[0, 2] / len(inputs),
        flat_output_accuracy=hits[1, 2] / len(inputs),
        corrupt_program_accuracy=hits[2, 2] / len(inputs),
        distilled_binding_accuracy=float(hits[0, :2].mean() / len(inputs)),
        commands=len(inputs),
        program_loads=int(inputs[:, 0].sum()),
    )
