import json
from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sonapse_recursion.targets import Circuit, Physical, reference
from sonapse_recursion.host import Host
from sonapse_recursion.emulation import train, study, structural
from sonapse_recursion.resources import Bank
from sonapse_recursion.substrate import Substrate
from sonapse_recursion.storage import save_host, load_host, save_circuit, load_circuit
from sonapse_recursion.runner import run
from sonapse_recursion import experiments, augments


class Dynamics(unittest.TestCase):
    def test_linear_rc(self):
        np.testing.assert_allclose(Physical("rc").step([1.0, 0.0], 0), [0.88, 0.08])

    def test_reference_lif(self):
        np.testing.assert_equal(Physical("lif").step([0.8, 0.0], 1.0), [0.0, 1.0])

    def test_signed_delayed_propagation(self):
        g = Circuit([[0, 0], [-1, 0]], [1, 0])
        s = Substrate(g, delay=2)
        self.assertEqual(s.step(1)[1], 0)
        self.assertEqual(s.step(0)[1], 0)
        self.assertLess(s.step(0)[1], 0)

    def test_model_family(self):
        g = Circuit.random(4, np.random.default_rng(7))
        for name in Substrate.models:
            with self.subTest(name=name):
                a, b = Substrate(g, name), Substrate(g, name)
                for drive in np.sin(np.arange(100)):
                    np.testing.assert_equal(a.step(drive), b.step(drive))
                self.assertTrue(np.isfinite(a.v).all())
                self.assertGreater(a.energy, 0)

    def test_refractory(self):
        s = Substrate(Circuit([[0]], [1]), "lif")
        self.assertEqual([s.step(2.0)[0] for _ in range(4)], [1, 0, 0, 1])

    def test_plasticity(self):
        s = Substrate(Circuit([[0.8]], [1]), plastic=True)
        for _ in range(10):
            s.step(0.5, 1.0)
        self.assertGreater(s.weights[0, 0], 0.8)

    def test_empty_and_disconnected(self):
        s = Substrate(Circuit(np.empty((0, 0)), []))
        self.assertEqual(len(s.step(1)), 0)
        np.testing.assert_equal(
            reference(Circuit([[0]], [0]), [1, 2]), np.zeros((3, 1))
        )

    def test_bad_architecture(self):
        for weights, ports in [([[1, 2]], [1]), ([[np.nan]], [1])]:
            with self.assertRaises(ValueError):
                Circuit(weights, ports)

    def test_no_teacher_in_rollout(self):
        host, _, _ = train(Circuit.random(2, np.random.default_rng(3)), 7)
        with patch.object(
            Circuit, "step", side_effect=AssertionError("reference leakage")
        ):
            self.assertEqual(host.rollout([0.3] * 30, np.zeros(2)).shape, (30, 2))

    def test_unstable_output_bounds(self):
        h = Host(1, 0, np.random.default_rng(0))
        h.readout = np.ones((3, 1)) * 100
        np.testing.assert_equal(h.rollout([1] * 4, np.zeros(1)), np.full((4, 1), 8))

    def test_autonomous_recurrent(self):
        result = study("circuit", 2, 5)
        self.assertLess(result["autonomous_rmse"], result["linear_rmse"])

    def test_structural_ablation(self):
        r = structural(3)
        self.assertLess(r["unseen_rmse"], r["shuffled_edges_rmse"])


class Resources(unittest.TestCase):
    def test_reserve_and_children(self):
        b = Bank(16, 4, 2)
        b.allocate("parent", 10)
        b.allocate("child", 4, "parent")
        b.allocate("sibling", 6, "parent")
        b.view("child")[:] = 2
        self.assertTrue((b.view("parent")[:4] == 2).all())
        self.assertEqual(b.state[:4].sum(), 0)
        with self.assertRaises(MemoryError):
            b.allocate("overflow", 3)
        with self.assertRaises(ValueError):
            b.allocate("nested", 1, "child")
        b.suspend("parent")
        self.assertFalse(b.enabled("child"))
        b.remove("parent")
        self.assertFalse(b.leases)
        self.assertEqual(b.state.sum(), 0)

    def test_invalid(self):
        b = Bank()
        with self.assertRaises(ValueError):
            b.allocate("bad", -1)
        with self.assertRaises(KeyError):
            b.allocate("orphan", 1, "missing")

    def test_pause_equivalence(self):
        self.assertEqual(augments.interruption(4)["resume_rmse"], 0)

    def test_graft_control(self):
        r = experiments.graft(1)
        self.assertLess(r["trained_rmse"], r["other_target_rmse"])

    def test_loss_and_prediction(self):
        r = experiments.channel(1)
        self.assertTrue(0 < r["packets_lost"] < r["packets_sent"])
        self.assertLess(r["predictive_rmse"], r["hold_rmse"])

    def test_evolution_inheritance(self):
        history = []
        r = experiments.evolution(2, history)
        self.assertGreater(r["accepted"], 0)
        self.assertTrue(
            all(a["score"] >= b["score"] for a, b in zip(history, history[1:]))
        )


class Persistence(unittest.TestCase):
    def test_mid_rollout_restore(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "host.json"
            host, _, _ = train(Physical("spring"), 0)
            x = host.rollout([0.1] * 10, np.zeros(2))[-1]
            save_host(path, host, x, 10)
            recovered, state, tick = load_host(path)
            self.assertEqual(tick, 10)
            np.testing.assert_equal(
                recovered.rollout([0.2] * 5, state), host.rollout([0.2] * 5, x)
            )
            bad = json.loads(path.read_text())
            bad["state"] = [0]
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_host(path)

    def test_graph_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            for n in (0, 4):
                path = Path(d) / "g.json"
                g = Circuit.random(n, np.random.default_rng(1))
                save_circuit(path, g)
                np.testing.assert_equal(g.weights, load_circuit(path).weights)

    def test_resume_interrupted_batch(self):
        with tempfile.TemporaryDirectory() as d:
            from sonapse_recursion.runner import execute

            def interrupted(pair):
                if pair[1] == 1:
                    raise RuntimeError("simulated interruption")
                return execute(pair)

            with (
                patch("sonapse_recursion.runner.execute", side_effect=interrupted),
                self.assertRaises(RuntimeError),
            ):
                run(["logic"], range(3), Path(d), 1)
            with closing(sqlite3.connect(Path(d) / "trials.sqlite")) as db:
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM trials").fetchone()[0], 1
                )
            first = run(["logic"], range(3), Path(d), 1)
            with patch("sonapse_recursion.runner.execute", side_effect=AssertionError):
                self.assertEqual(first, run(["logic"], range(3), Path(d), 1))

    def test_repeat_seed(self):
        self.assertEqual(experiments.channel(89), experiments.channel(89))
