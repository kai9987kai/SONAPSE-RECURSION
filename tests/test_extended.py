import unittest
import numpy as np
from sonapse_recursion.domains import Domain
from sonapse_recursion.targets import reference, Circuit
from sonapse_recursion.emulation import train
from sonapse_recursion.experiments import channel
from sonapse_recursion.grafts import Grafts, Specification
from sonapse_recursion.resources import Bank
from sonapse_recursion.sparse import EventConnectome
from sonapse_recursion.discovery import curriculum, pareto, closed_loop


class Extended(unittest.TestCase):
    def test_reference_catalog(self):
        for kind, n in Domain.sizes.items():
            with self.subTest(kind=kind):
                trace = reference(Domain(kind), np.sin(np.arange(512)))
                self.assertEqual(trace.shape, (513, n))
                self.assertTrue(np.isfinite(trace).all())

    def test_loss_and_delay_extremes(self):
        self.assertEqual(channel(0, 0, 0, 0)["hold_rmse"], 0)
        lost = channel(0, 1)
        self.assertEqual(lost["packets_lost"], 640)
        self.assertEqual(lost["predictive_rmse"], lost["hold_rmse"])
        with self.assertRaises(ValueError):
            channel(0, 2)

    def test_sparse_cancellation(self):
        events = EventConnectome(4, [(0, 2, 1, 2), (1, 2, -1, 2), (0, 3, 1, 4)])
        events.inject(0, 0, 1)
        events.inject(0, 1, 1)
        self.assertEqual(events.advance(3), [(0, 0), (0, 1)])
        self.assertEqual(events.advance(5), [(4, 3)])
        self.assertEqual(events.advance(1000), [])

    def test_sparse_loop_and_invalid(self):
        events = EventConnectome(1, [(0, 0, 1, 2)])
        events.inject(0, 0, 1)
        self.assertEqual(events.advance(5), [(0, 0), (2, 0), (4, 0)])
        with self.assertRaises(ValueError):
            events.inject(1, 0, 1)
        with self.assertRaises(ValueError):
            EventConnectome(1, [(1, 0, 1, 1)])

    def test_graft_schedule(self):
        host, _, _ = train(Circuit.random(2, np.random.default_rng(1)), 2)
        bank = Bank(16, 2)
        bank.allocate("container", 8)
        runtime = Grafts(bank)
        runtime.install(Specification("worker", 2, "container"), host)
        first = runtime.communicate("worker", 0.6, 0)
        np.testing.assert_equal(first, runtime.communicate("worker", -1, 1))
        bank.suspend("container")
        np.testing.assert_equal(first, runtime.communicate("worker", -1, 2))
        bank.suspend("container", False)
        runtime.communicate("worker", -0.5, 4)
        self.assertEqual(runtime.describe("worker")["updates"], 2)
        self.assertEqual(
            runtime.multiply_adds, 2 * (host.projection.size + host.readout.size)
        )
        runtime.remove("worker")
        self.assertEqual(bank.view("container").sum(), 0)

    def test_invalid_period(self):
        host, _, _ = train(Circuit.random(1, np.random.default_rng(1)), 2)
        with self.assertRaises(ValueError):
            Grafts(Bank()).install(Specification("test", 0), host)

    def test_search_is_deterministic(self):
        self.assertEqual(pareto(3), pareto(3))
        self.assertIn(curriculum(2)["largest_passed"], (0, 1, 2, 5, 10, 25))

    def test_closed_loop(self):
        for value in closed_loop(3).values():
            self.assertTrue(np.isfinite(value))
