"""Scheduled virtual ports backed by a fixed state bank and trained host kernels."""

from dataclasses import dataclass
import numpy as np
from .host import Host
from .resources import Bank


@dataclass(frozen=True)
class Specification:
    name: str
    period: int = 1
    parent: str | None = None
    verification: str = "experimental"


class Grafts:
    def __init__(self, bank: Bank):
        self.bank = bank
        self.instances = {}
        self.updates = {}
        self.multiply_adds = 0

    def install(self, spec: Specification, kernel: Host):
        if (
            not spec.name
            or type(spec.period) is not int
            or spec.period < 1
            or kernel.size < 1
            or kernel.readout is None
            or not np.isfinite(kernel.readout).all()
        ):
            raise ValueError("Incompatible graft")
        if spec.parent is not None and spec.parent in self.instances:
            raise ValueError("A running kernel owns all coordinates in its lease")
        self.bank.allocate(spec.name, kernel.size, spec.parent)
        self.instances[spec.name] = (spec, kernel)
        self.updates[spec.name] = 0

    def communicate(self, name, drive, tick):
        spec, kernel = self.instances[name]
        if not np.isfinite(drive) or type(tick) is not int or tick < 0:
            raise ValueError("Invalid message")
        state = self.bank.view(name)
        if tick % spec.period == 0 and self.bank.enabled(name):
            state[:] = kernel.step(state, drive)
            self.multiply_adds += kernel.projection.size + kernel.readout.size
            self.updates[name] += 1
        return state.copy()

    def remove(self, name):
        self.bank.remove(name)
        del self.instances[name]
        del self.updates[name]

    def describe(self, name):
        spec, kernel = self.instances[name]
        return dict(
            name=name,
            execution="trained recurrent transition",
            period=spec.period,
            parent=spec.parent,
            scalar_inputs=1,
            vector_outputs=kernel.size,
            state_cells=kernel.size,
            weights=kernel.projection.size + kernel.readout.size,
            updates=self.updates[name],
            enabled=self.bank.enabled(name),
            plasticity="frozen at deployment",
            verification=spec.verification,
        )
