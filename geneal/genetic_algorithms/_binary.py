from datetime import timedelta
from typing import Sequence

import numpy as np

from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver


class BinaryGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        n_bits: int = 1,
        max_gen: int = None,
        max_time: timedelta = None,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = True,
        show_stats: bool = True,
        plot_results: bool = True,
        excluded_genes: Sequence = None,
        n_crossover_points: int = 1,
        random_state: int = None,
        checkpoint_folder: str = None,
    ):
        """
        :param fitness_function: can either be a fitness function or a class implementing a fitness function +
        methods to override the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param n_bits: number of bits representing each gene
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        :param selection_strategy: strategy to use for selection
        :param verbose: whether to print iterations status
        :param show_stats: whether to print stats at the end
        :param plot_results: whether to plot results of the run at the end
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes * n_bits,
            max_gen=max_gen,
            max_time=max_time,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate,
            selection_strategy=selection_strategy,
            verbose=verbose,
            show_stats=show_stats,
            plot_results=plot_results,
            excluded_genes=excluded_genes,
            n_crossover_points=n_crossover_points,
            random_state=random_state,
            checkpoint_folder=checkpoint_folder,
        )

    def checkpoint(self, best_individual, best_fitness, gen_n, fitness, population):
        """
        Callback for a child class to do whatever they want with the data presented.
        They may save, print, or do nothing on these checkpoints.

        :param best_individual: the current individual with the highest fitness
        :param gen_n: the number of which generation the loop is at
        :param fitness: the array of the fitness values of the entire population
        :param population: the entire population at that point
        """
        pass

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and outputs individuals
        composed of 0's and 1's.

        :return: a numpy array with a randomized initialized population
        """

        bits_array = []

        for _ in range(self.pop_size):
            bits = np.zeros(self.n_genes)
            bits[: np.random.randint(0, self.n_genes)] = 1
            np.random.shuffle(bits)

            bits_array.append(bits)

        return np.array(bits_array)

    @staticmethod
    def create_offspring(first_parent, sec_parent, crossover_pt, _):
        """
        Creates an offspring from 2 parents. It performs the crossover
        according the following rule:

        offspring = [first_parent[:crossover_pt], sec_parent[crossover_pt:]
        """

        return np.hstack(
            (first_parent[: crossover_pt[0]], sec_parent[crossover_pt[0] :])
        )

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        mutation_rows, mutation_cols = super(
            BinaryGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_cols] = np.abs(population - 1)[
            mutation_rows, mutation_cols
        ]

        return population
