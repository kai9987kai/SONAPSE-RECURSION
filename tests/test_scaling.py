"""Regression checks for sparse event ordering and chunked paired-bootstrap reports."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sonapse_recursion.runner import run
from sonapse_recursion.sparse import EventConnectome


class ScaleTests(unittest.TestCase):
    def test_out_of_order_events_merge_at_the_same_tick(self):
        net = EventConnectome(2, [(0, 1, 1, 50)], refractory=0)
        net.inject(80, 0, 0.5)
        net.inject(2, 0, 1)
        net.inject(80, 0, 0.5)
        self.assertEqual(net.advance(52), [(2, 0)])
        self.assertEqual(net.advance(81), [(52, 1), (80, 0)])
        self.assertEqual(net.advance(131), [(130, 1)])
        self.assertEqual(net.advance(9999), [])

    def test_chunked_intervals_match_full_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run(["channel"], range(70), Path(directory), 1)
            trial = report["studies"]["channel"]
            draws = np.random.default_rng(491).integers(70, size=(2000, 70))
            values = np.array([x["metrics"]["hold_rmse"] for x in trial["trials"]])
            expected = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
            np.testing.assert_array_equal(trial["bootstrap_95"]["hold_rmse"], expected)
            subset = run(["channel"], range(2, 4), Path(directory), 1)
            self.assertEqual(
                subset["studies"]["channel"]["trials"], trial["trials"][2:4]
            )
