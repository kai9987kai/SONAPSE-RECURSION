import tempfile
import unittest
from pathlib import Path
import numpy as np
from sonapse_recursion.import_connectome import import_csv
from sonapse_recursion.governor import Budget, Governor, Job
from sonapse_recursion.river import river_load, episode


class FieldTests(unittest.TestCase):
    def test_direction_and_import_filter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "edges.csv"
            p.write_text(
                "Source,Target,Weight,Type\nA,B,7,chemical\nB,A,2,electrical\n"
            )
            graph, nodes, sha = import_csv(p, 2)
            self.assertEqual(nodes, ["A", "B"])
            self.assertEqual(len(sha), 64)
            self.assertGreater(graph.weights[1, 0], 0)
            self.assertEqual(graph.weights[0, 1], 0)
            p.write_text("Source,Target,Weight,Type\nA,B,-3,chemical\n")
            with self.assertRaises(ValueError):
                import_csv(p, 2)

    def test_governor_interrupt_and_reservation(self):
        one = Budget(1, 1, 1, 1, 1)
        governor = Governor(Budget(2, 2, 2, 2, 2), one)
        jobs = [
            Job("sim", one, 3, 1, 6, 1),
            Job("extra", one, 3, 1, 6, 1),
            Job("native", one, 3, 1, 6, 1, True),
        ]
        self.assertEqual(governor.dispatch(jobs, 0, "fifo"), ["sim", "native"])
        self.assertEqual(governor.dispatch(jobs, 1, "fifo", True), ["native"])
        self.assertEqual(jobs[0].remaining, 2)
        self.assertEqual(governor.dispatch(jobs, 7, "fifo"), [])

    def test_river_cost_and_usefulness(self):
        self.assertGreater(river_load(0.8, 0.05, 1), river_load(0.8, -0.05, 1))
        # Water is high now but rapidly falls. The accurate planner can use this crossing.
        data = [(0.85, -0.07, 3, 0.0, 0.0, np.zeros(3, dtype=bool))] * 12
        self.assertGreater(
            episode(data, "unrestricted")[0], episode(data, "reactive")[0]
        )
        self.assertEqual(episode(data, "external")[3], 36)
