"""
Tests to check the functioning of the assertion errors in the GeneticAlgorithm class
"""
from preferendus import Preferendus
import unittest


def objective(x):
    return x * 10


class TestTripwires(unittest.TestCase):
    """
    https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-
    throws-an-exception
    """

    working_constraints = (("ineq", lambda a: a + 10),)
    working_bounds = ((1, 10), (2, 20))

    def test_callable(self):
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=1,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
            )

    def test_cons(self):
        with self.assertRaises(AssertionError) as context:
            ga = Preferendus(
                objective=objective,
                constraints=(("foo", lambda a: a + 10),),
                bounds=self.working_bounds,
            )
            ga.run(verbose=False)

    def test_pop(self):
        options = {"n_pop": 3}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_r_cross(self):
        options = {"r_cross": 3}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_type_definition(self):
        options = {"var_type": "int", "var_type_mixed": ["real", "int"]}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_wrong_type_definition(self):
        options = {"var_type": "foo"}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

        options = {"var_type_mixed": ["foo", "foo"]}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_lengths_bounds_var_types(self):
        options = {"var_type_mixed": ["foo", "foo", "bar"]}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_wrong_agg_definition(self):
        options = {"aggregation": "foo"}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_args(self):
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                args=[],
            )

    def test_elitism(self):
        options = {"elitism percentage": 200}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_mutation_order1(self):
        options = {"mutation_rate_order": 6}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )

    def test_mutation_order2(self):
        options = {"mutation_rate_order": -1}
        with self.assertRaises(AssertionError) as context:
            Preferendus(
                objective=objective,
                constraints=self.working_constraints,
                bounds=self.working_bounds,
                options=options,
            )


if __name__ == "__main__":
    unittest.main()
