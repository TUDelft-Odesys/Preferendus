"""
This algorithm is build by Harold van Heukelum, with special focus on the applicability
for a priori preference function modeling in combination with Tetra
(www.scientificmetrics.com). The sources listed below are used in the creation of this
algorithm.

For mutation determination, see [smith, 2015]

Sources:
    Brownlee, J. (2021, March 3). Simple genetic algorithm from scratch in Python.
    Machine Learning Mastery. Retrieved November 25, 2021, from
    https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/.

    Kramer, O. (2008). Self-adaptive heuristics for evolutionary computation. Springer.

    Solgi, R. M. (2020). geneticalgorithm: Genetic algorithm package for Python. GitHub.
    Retrieved April 20, 2022, from https://github.com/rmsolgi/geneticalgorithm

Copyright (c) 2022. Harold Van Heukelum
"""
from time import perf_counter
from typing import Union

import numpy as np
from numpy.random import randint, normal

from ._constraints import _const_handler
from ._decoder import _Decoding
from ._nextgen import _selection, _mutation, _crossover
from .tetra_pfm import TetraSolver
from .weighted_minmax import aggregate_max


class _Colors:
    """Class to allow for printing in color on the console"""

    OK = "\033[92m"  # GREEN
    WARNING = "\033[93m"  # YELLOW
    FAIL = "\033[91m"  # RED
    RESET = "\033[0m"  # RESET COLOR


class Preferendus:
    """
    Optimization algorithm based on the Open Design Systems methodology.
    """

    def __init__(
        self,
        objective,
        constraints: Union[tuple, list],
        bounds: Union[tuple, list],
        cons_handler: str = "simple",
        options: dict = None,
        args: tuple = None,
        start_points_population: list = None,
    ):
        """
        The following parameters can be defined in the option dictionary:
            n_bits: Number of bits per variable, only relevant for non-integer
                var_types. See docstring of decode function for details on
                specification of number of bits.

            n_iter: Maximum number of generations before algorithm is stopped. Prevents
                infinite runtimes.

            n_pop: Size of the population that is generated per generation, ie. number
                of guesses per generation

            r_cross: Value between 0 and 1 that determine how often crossover is
                performed.

            max_stall: Number of generations that must have a change < tol before the
                algorithm is stopped.

            tol: tolerance for change. If improvement is below this value, it is
                handled as no change.

            var_type: Type of variables that are considered. For integer, use 'int'.
                Else, use 'real'. Sets the type for all the variables. For
                mixed-integer problems, use var_type_mixed.

            var_type_mixed: tuple with the types of the variables. For integer, use
            'int'; for bool, use 'bool; else, use 'real'.

            aggregation: the GA can be used as an ordinary GA by setting the
                aggregation to None (default). To call the Preferendus, set this value
                to either 'IMAP' or 'minmax' (depending on the method you want to use).

            mutation_rate_order: The mutation can be influenced via this parameter
                (default: 2).

            elitism percentage: this parameter influences the percentage of the
                population that is considered in the elitism (default: 15%)


        :param objective: the function to minimize. Must be in the form f(x, *args)
            with x is a 2-D array of width len(bounds) and length n_pop
        :param constraints: tuple of constraint functions (format: [[type, func]])
        :param bounds: boundaries for variables in x. Every variable in x should have a
            boundary!
        :param cons_handler: simple (default) or CND (Coello non-dominance)
        :param options: dictionary that contains all parameters for the GA. See
            docstring for explanation of these parameters
        :param start_points_population: list containing the starting points for the GA
            (when given, otherwise a random population is created)
        :return: list with best bitstring, the optimal result of the objective
            function, and the scores for the variables in x
        """
        if options is None:
            options = {}
        self.n_bits: int = options.get("n_bits", 12)
        self.n_iter: int = options.get("n_iter", 500)
        self.n_pop: int = options.get("n_pop", 120)
        self.r_cross: float = options.get("r_cross", 0.85)
        self.max_stall: int = options.get("max_stall", 100)
        self.tol: float = options.get("tol", 1e-7)
        self.var_type: str = options.get("var_type", None)
        self.var_type_mixed = options.get("var_type_mixed", None)
        self.aggregation = options.get("aggregation", "IMAP")
        self.elitism_percentage = options.get("elitism percentage", 10) / 100
        self.mutation_rate_order = options.get("mutation_rate_order", 2)

        ################################################################################
        # assert if important input are correct
        assert callable(objective), "Objective must be callable"
        self.objective = objective

        for eq, func in constraints:
            assert (
                eq == "eq" or eq == "ineq"
            ), "Type of constraint should be 'eq' or 'ineq'"
            assert callable(func), "Constraint function must be callable"
        self.constraints = constraints
        self.cons_handler = cons_handler

        assert self.n_pop % 2 == 0, "N_pop must be even"

        assert 0 < self.r_cross < 1, "Crossover rate r_cross should be between 0 and 1"

        assert (
            self.var_type is None or self.var_type_mixed is None
        ), "Var_type and var_type_mixed cannot both be set"

        assert self.var_type in ["int", "bool", "real"] or self.var_type is None, (
            "Variable type (var_type) " "must be 'int', 'bool' or 'real'"
        )

        if self.var_type_mixed is not None:
            for item in self.var_type_mixed:
                assert item in ["int", "bool", "real"], (
                    "Type of variable in var_type_mixed must be 'int', 'bool or "
                    "'real'"
                )

        assert self.aggregation in [
            None,
            "IMAP",
            "minmax",
        ], "Wrong aggregation, should be tetra, minmax, or None (default)"
        print(
            _Colors.WARNING
            + f"The type of aggregation is set to {self.aggregation}"
            + _Colors.RESET
        )

        assert isinstance(args, tuple) or args is None, "Args must be of type tuple"
        if args is None:
            self.args = tuple()
        else:
            self.args = args

        assert (
            0 <= self.elitism_percentage <= 1
        ), "Elitism percentage must be given as a value between 0 and 100"
        self.start_point_population = start_points_population

        assert (
            0 < self.mutation_rate_order < 5
        ), "An mutation rate order < 0 or > 5 is not tested."

        ################################################################################
        # create list with variable types
        if self.var_type is None:
            if self.var_type_mixed is None:
                approach = list(["real"] * len(bounds))
            else:
                approach = self.var_type_mixed
        else:
            approach = list([self.var_type] * len(bounds))
        self.approach = approach

        ################################################################################
        self.bounds = bounds

        self.solver = TetraSolver()
        self.dec = _Decoding(
            bounds=self.bounds, n_bits=self.n_bits, approach=self.approach
        )

    def _initiate_population(self) -> tuple[list, int]:
        """
        Method to create the initial population, based on the self.approach list
        :return: initial population, r_count measure
        """
        r_count = 0
        pop = list([0] * self.n_pop)
        for p in range(len(pop)):
            solo = list([0] * len(self.bounds))
            for i in range(len(solo)):
                if self.approach[i] == "int":
                    solo[i] = randint(self.bounds[i][0], self.bounds[i][1] + 1)
                    r_count += 1
                elif self.approach[i] == "bool":
                    solo[i] = randint(0, 2)
                    r_count += 1
                else:
                    solo[i] = randint(0, 2, self.n_bits).tolist()
                    r_count += self.n_bits
            pop[p] = solo.copy()
        return pop, r_count

    def _starting_points_population(self, starting_points: list) -> list:
        """
        Method to create the initial population based on provided starting points
        :return:
        """
        floor_res = np.floor_divide(self.n_pop, len(starting_points))
        res = self.n_pop % len(starting_points)

        pop = starting_points * floor_res
        for i in range(res):
            pop.append(starting_points[i])

        return pop

    def rerun_eval(
        self,
        scores: Union[list, np.ndarray],
        decoded: Union[list, np.ndarray],
    ) -> Union[list, object]:
        """
        Method to rerun the evaluation of the population without the worst scoring
        members.
        :param scores: original scores of the population
        :param decoded: decoded population
        :return: new scores
        """
        scores_arr = np.array(scores)
        decoded_arr = np.array(decoded)
        mask_too_low = scores_arr < -20.0  # considering only score between -100 and -20

        if len(decoded_arr[mask_too_low]) > 1:
            w_masked, p_masked = self.objective(decoded_arr[mask_too_low], *self.args)
            scores_masked = self.solver.request(w_masked, p_masked)
            scores_arr[mask_too_low] = scores_masked
            scores_arr[np.invert(mask_too_low)] = 0

        return scores_arr.tolist()

    def _runner(self, pop, r_count, verbose):
        """

        :return:
        """
        # set initial best and best_score
        best_score = 1e12
        best = pop[
            randint(0, len(pop))
        ]  # select random member of pop as initial best guess
        scores_iter = []

        # set initial parameters
        stall_counter = 0
        gen = 0
        n_infeasible = 0
        check_array_complete = list()

        # set mutation rate
        t = 1 / r_count ** (1 / self.mutation_rate_order)
        r_mut = t

        # loop through generations
        for gen in range(self.n_iter):
            best_eval_old = best_score

            # decode population. Should be np.array to make masks possible
            decoded = np.array([self.dec.decode(p) for p in pop])

            # check diversity:
            diversity = round(
                np.max(np.unique(decoded, return_counts=True)[1])
                / (len(pop) * len(pop[0])),
                3,
            )

            # evaluate all candidates in the population
            if self.aggregation == "IMAP":
                w, p = self.objective(decoded, *self.args)
                scores = self.solver.request(w, p)
                scores = self.rerun_eval(scores, decoded)

            elif self.aggregation == "minmax":
                w, p = self.objective(decoded, *self.args)
                scores = aggregate_max(w, p, 100)

            else:
                scores = self.objective(decoded, *self.args)

            scores, n_infeasible = _const_handler(
                self.cons_handler, self.constraints, decoded, scores
            )

            def print_status():
                """
                Function to print progress to console
                """
                if verbose:
                    print(
                        "{:<12} {:<12} {:<16} {:<12} {:<12} {:<12}".format(
                            gen,
                            round(best_score, 4),
                            round(float(np.median(scores)), 4),
                            stall_counter,
                            diversity,
                            n_infeasible,
                        )
                    )

            # check for new best solution
            if self.aggregation == "IMAP":
                check_array_complete.append(decoded[np.argmin(scores)].tolist())

                w, p = self.objective(np.array(check_array_complete), *self.args)
                result = self.solver.request(w, p)

                result = self.rerun_eval(result, check_array_complete)

                assert len(check_array_complete) == gen + 1, (
                    f"Error: len check_array {len(check_array_complete)} != "
                    f"gen nr + 1 {gen + 1}"
                )

                if result[-1] <= min(result):
                    best_score = min(scores)
                    best = pop[np.argmin(scores)]
                    scores_iter.append(best_score)

                result = np.array(np.round(result, 3))

                if -100.0 in result:
                    counter_1 = np.count_nonzero(result == min(result))
                    counter_2 = len(result) - np.argmin(scores)
                    stall_counter = max(counter_1, counter_2)
                elif np.all(result == -50.0):
                    stall_counter = np.count_nonzero(result == -50.0)
                else:
                    stall_counter = 1

            else:  # normal GA evaluation
                mask = np.array(scores) < best_score
                if mask.any():
                    stall_counter = 0
                    best_score = min(scores)
                    best = pop[np.argmin(scores)]
                    scores_iter.append(best_score)

                if abs(best_eval_old - best_score) < self.tol:
                    stall_counter += 1

            print_status()
            if stall_counter >= self.max_stall:
                if verbose:
                    print(f"Stopped at gen {gen}")
                break

            # determine the part that is elite
            unique_scores = np.unique(scores)
            n_elite = (
                int(self.elitism_percentage * len(unique_scores))
                - int(self.elitism_percentage * len(unique_scores)) % 2
            )

            children = list()
            if n_elite != 0:
                elites = unique_scores[-1 * n_elite :]
                for score in elites:
                    for i, p in enumerate(pop):
                        if scores[i] == score:
                            children.append(p.copy())
                            break

            # select parents for rest of new generation
            selected = [_selection(pop, scores) for _ in range(self.n_pop - n_elite)]

            r_mut = r_mut * np.exp(t * normal(0, 1))
            for i in range(0, len(selected), 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]

                # crossover and mutation
                for c in _crossover(p1, p2, self.r_cross, self.approach):
                    # mutation
                    _mutation(c, r_mut, self.approach, self.bounds)

                    # store for next generation
                    children.append(c)

            # replace population
            pop = children
            assert len(pop) == self.n_pop, (
                f"Pop array is not equal after children are made. "
                f"It is now {len(pop)} and should be n_pop = {self.n_pop}"
            )

        if gen == self.n_iter - 1:
            print(
                _Colors.FAIL
                + "The iteration is stopped since the max number is reached. The "
                "results might be incorrect! Please be cautious." + _Colors.RESET
            )
        elif gen < 1.1 * self.max_stall:
            print(
                _Colors.FAIL
                + "The number of generations is terribly close to the number of max "
                "stall iterations. This suggests a too fast convergence and wrong "
                "results."
            )
            print(
                "Please be careful in using these results and assume they are wrong "
                "unless proven otherwise!" + _Colors.RESET
            )

        decoded = self.dec.decode(best)  # get final values for variables
        return best_score, decoded, scores_iter, n_infeasible

    def run(self, verbose: bool = True) -> tuple[float, Union[list, np.ndarray], list]:
        """
        Run the genetic algorithm

        :param verbose: allow printing to console (True by default)
        :return: the best evaluation; the best member of population; scores per gen
        """
        tic = perf_counter()

        ################################################################################
        # create initial (random) population
        pop, r_count = self._initiate_population()

        ################################################################################
        # run optimization

        # print headers for console output of algorithm
        if verbose:
            print(
                "{:<12} {:<12} {:<16} {:<12} {:<12} {:<12}".format(
                    "Generation",
                    "Best score",
                    "Median",
                    "Max stall",
                    "Diversity",
                    "Number of non-feasible results",
                )
            )

        if self.aggregation == "IMAP":
            if self.start_point_population is None:
                print(
                    _Colors.WARNING
                    + "No initial starting point for the optimization with tetra is "
                    "given. A random population is generated." + _Colors.RESET
                )
            else:
                starting_points = [
                    self.dec.inverse_decode(member)
                    for member in self.start_point_population
                ]
                pop = self._starting_points_population(starting_points)

            best_score, decoded, scores_iter, n_infeasible = self._runner(
                pop, r_count, verbose
            )
        else:
            best_score, decoded, scores_iter, n_infeasible = self._runner(
                pop, r_count, verbose
            )

        ################################################################################
        # print timer
        toc = perf_counter()
        assert n_infeasible < self.n_pop, "No feasible solutions could be found"

        if verbose:
            print(f"Execution time was {toc - tic:0.4f} seconds")

        return best_score, decoded, scores_iter
