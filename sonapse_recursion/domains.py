"""Additional reference tasks sharing the same scalar-drive transition protocol."""

import numpy as np


class Domain:
    sizes = {
        "oscillator": 2,
        "motor": 2,
        "memory": 2,
        "associative": 3,
        "rlc": 2,
        "ballistic": 2,
        "linkage": 4,
        "dam_load": 3,
        "register": 2,
        "automaton": 7,
    }

    def __init__(self, kind):
        self.kind = kind
        self.size = self.sizes[kind]

    def step(self, state, drive):
        x, y = state[:2]
        if self.kind == "oscillator":
            radius = x * x + y * y
            return np.array(
                [
                    x + 0.08 * ((1 - radius) * x - y + drive),
                    y + 0.08 * ((1 - radius) * y + x),
                ]
            )
        if self.kind == "motor":
            torque = np.tanh(2 * (drive - x) - 0.6 * y)
            return np.array([x + 0.1 * y, 0.96 * y + 0.1 * torque])
        if self.kind == "memory":
            return np.array([drive if abs(drive) > 0.6 else x, 0.85 * y + 0.15 * x])
        if self.kind == "associative":
            pattern = np.array([1.0, -1.0, 1.0])
            return np.tanh(pattern * (pattern @ state) / 2 + drive)
        if self.kind == "rlc":
            current = y + 0.09 * (drive - 0.3 * y - x)
            return np.array([x + 0.09 * current, current])
        if self.kind == "ballistic":
            velocity = y + 0.05 * (drive - 0.7)
            height = x + 0.05 * velocity
            return np.array(
                [max(0, height), -0.7 * velocity if height < 0 else velocity]
            )
        if self.kind == "linkage":
            a, b = state[2:]
            b += 0.08 * (-np.sin(a) + 0.25 * (x - a) - 0.15 * b)
            y += 0.08 * (-np.sin(x) + 0.25 * (a - x) - 0.15 * y + drive)
            return np.array([x + 0.08 * y, y, a + 0.08 * b, b])
        if self.kind == "dam_load":
            rise = max(0, drive) * 0.03
            flux = max(0, x - y - 0.1) * 0.07
            return np.array(
                [
                    0.98 * x + rise - flux,
                    0.96 * y + flux,
                    0.85 * state[2] + 0.15 * (x - y) ** 2,
                ]
            )
        if self.kind == "register":
            return np.array([float(drive > 0), float(x > 0.5)])
        bits = state > 0.5
        nxt = np.logical_xor(np.roll(bits, 1), np.roll(bits, -1))
        nxt[0] ^= drive > 0.5
        return nxt.astype(float)
