"""Tests for the tpt_opt extension module. Run with: python -m unittest discover -s tests"""

import math
import unittest

import tpt_opt


class ProblemTests(unittest.TestCase):
    def test_knapsack_maximum(self):
        p = tpt_opt.Problem()
        x = [p.add_variable(kind="binary") for _ in range(3)]
        self.assertEqual(x, [0, 1, 2])
        p.add_constraint([(0, 4.0), (1, 3.0), (2, 2.0)], upper=7.0)
        p.set_objective([(0, 5.0), (1, 4.0), (2, 3.0)], sense="maximize")
        sol = p.solve()
        self.assertEqual(sol.status, "optimal")
        self.assertAlmostEqual(sol.objective, 9.0, places=6)
        self.assertEqual([round(v) for v in sol.primal], [1, 1, 0])

    def test_objective_constant_is_included(self):
        p = tpt_opt.Problem()
        y = p.add_variable(lower=2.0, upper=10.0)
        p.set_objective([(y, 3.0)], constant=1.5)
        sol = p.solve()
        self.assertAlmostEqual(sol.objective, 7.5, places=6)

    def test_infeasible_status(self):
        p = tpt_opt.Problem()
        y = p.add_variable(lower=0.0, upper=1.0)
        p.add_constraint([(y, 1.0)], lower=5.0)
        p.set_objective([(y, 1.0)])
        self.assertEqual(p.solve().status, "infeasible")

    def test_text_round_trip_keeps_the_optimum(self):
        p = tpt_opt.Problem()
        a = p.add_variable(kind="integer", lower=0.0, upper=6.0)
        b = p.add_variable(kind="integer", lower=0.0, upper=6.0)
        p.add_constraint([(a, 2.0), (b, 3.0)], upper=12.0)
        p.set_objective([(a, 3.0), (b, 4.0)], sense="maximize")
        before = p.solve().objective
        for text, loader in [(p.to_lp(), tpt_opt.Problem.from_lp), (p.to_mps(), tpt_opt.Problem.from_mps)]:
            again = loader(text)
            self.assertEqual(again.num_vars, 2)
            self.assertEqual(again.num_constraints, 1)
            self.assertAlmostEqual(again.solve().objective, before, places=6)

    def test_bad_inputs_raise_value_error(self):
        p = tpt_opt.Problem()
        p.add_variable()
        with self.assertRaises(ValueError):
            p.add_variable(kind="bogus")
        with self.assertRaises(ValueError):
            p.add_constraint([(5, 1.0)], upper=1.0)
        with self.assertRaises(ValueError):
            p.set_objective([(0, 1.0)], sense="sideways")
        with self.assertRaises(ValueError):
            tpt_opt.Problem.from_lp("Minimize\n obj: x\nSubject To\n r: x >= \nEnd\n")

    def test_seeded_parallel_solve_matches_sequential(self):
        p = tpt_opt.Problem()
        xs = [p.add_variable(kind="binary") for _ in range(8)]
        weights = [3.0, 5.0, 2.0, 7.0, 4.0, 6.0, 1.0, 8.0]
        values = [4.0, 6.0, 3.0, 9.0, 5.0, 7.0, 2.0, 10.0]
        p.add_constraint(list(zip(xs, weights)), upper=20.0)
        p.set_objective(list(zip(xs, values)), sense="maximize")
        one = p.solve(threads=1, seed=3).objective
        four = p.solve(threads=4, seed=3).objective
        self.assertAlmostEqual(one, four, places=6)


class AnnealingTests(unittest.TestCase):
    def test_finds_the_minimum_of_a_bowl(self):
        def bowl(x):
            return (x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2

        best_x, best_value = tpt_opt.simulated_annealing(bowl, [(-1.0, 1.0), (-1.0, 1.0)], seed=1, iterations=5000)
        self.assertLess(best_value, 1e-2)
        self.assertTrue(math.isfinite(best_x[0]))

    def test_same_seed_gives_the_same_result(self):
        def f(x):
            return sum(v * v for v in x)

        a = tpt_opt.simulated_annealing(f, [(-2.0, 2.0)] * 3, seed=9, iterations=800)
        b = tpt_opt.simulated_annealing(f, [(-2.0, 2.0)] * 3, seed=9, iterations=800)
        self.assertEqual(a, b)

    def test_exception_in_the_objective_is_reraised(self):
        def broken(x):
            raise KeyError("boom")

        with self.assertRaises(KeyError):
            tpt_opt.simulated_annealing(broken, [(0.0, 1.0)], seed=1, iterations=10)


class MultiObjectiveTests(unittest.TestCase):
    def problem(self):
        # f1 = x0 + x1 and f2 = -x0 + x1 over the unit square.
        return tpt_opt.LinearMultiObjective(
            [([1.0, 1.0], 0.0), ([-1.0, 1.0], 0.0)],
            [(0.0, 1.0), (0.0, 1.0)],
        )

    def test_weighted_sum_picks_the_weighted_optimum(self):
        x, f = self.problem().weighted_sum([1.0, 0.0])
        self.assertAlmostEqual(f[0], 0.0, places=6)
        self.assertAlmostEqual(f[1], 0.0, places=6)

    def test_epsilon_constraint_respects_the_bound(self):
        # Minimise f2 subject to f1 <= 1.0; the best f2 is -1 at x = (1, 0), which
        # gives f1 = 1 and satisfies the bound exactly.
        x, f = self.problem().epsilon_constraint(1, [1.0, 0.0])
        self.assertLessEqual(f[0], 1.0 + 1e-6)
        self.assertAlmostEqual(f[1], -1.0, places=6)

    def test_payoff_table_gives_the_ideal_point(self):
        ideal, nadir = self.problem().payoff_table()
        self.assertAlmostEqual(ideal[0], 0.0, places=6)
        self.assertAlmostEqual(ideal[1], -1.0, places=6)
        self.assertGreaterEqual(nadir[0], ideal[0])

    def test_nsga2_returns_mutually_non_dominated_points(self):
        front = self.problem().nsga2(population=40, generations=60, seed=5)
        self.assertGreater(len(front), 1)

        def dominates(a, b):
            # Same 1e-9 tolerance as the library: near-equal values are not better.
            return all(p <= q + 1e-9 for p, q in zip(a, b)) and any(p < q - 1e-9 for p, q in zip(a, b))

        for i, (_, a) in enumerate(front):
            for j, (_, b) in enumerate(front):
                if i != j:
                    self.assertFalse(dominates(a, b), f"{a} dominates {b}")

    def test_coefficient_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            tpt_opt.LinearMultiObjective([([1.0], 0.0)], [(0.0, 1.0), (0.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
