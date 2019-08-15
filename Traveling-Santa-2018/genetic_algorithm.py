import numpy as np

from inspect import currentframe, getargvalues

from chromosome import RudolphRoute


class RudolphEvolution(object):
    def __init__(
        self, current_best_length=None, precursors=200, chosen_size=20,
        generations=500, mutation_proba=.01, mutation_dimming_factor=1,
        verbose=0
    ):
        icf = currentframe()
        args, _, _, values = getargvalues(icf)
        values.pop('self')

        for param, value in values.items():
            setattr(self, param, value)

    def __call__(self):
        pass


def start_evolution(
    current_best_length=None, precursors=200, chosen_size=20, generations=500,
    mutation_proba=.01, mutation_dimming_factor=1, verbose=1
):
    """Initializes the genetic algorithm that solves the Santa-TSP problem.

    Parameters
    ----------
    current_best_length: float
        Length of the best route found so far. If not specified then takes
        the shortest route from the first generated population.

    precursors: int
        Size of the initiating population.

    chosen_size: float or int
        If passed as integer defines how many chromosomes to choose. If passed
        as float defines what fraction of the population to choose.

    generations: int
        Number of iterations that algorithm will do trying to find the
        chromosome with the highest fitness.

    Returns
    -------
    alpha_chrom: RudolphRoute
        Object of the RudolphRoute class with the shortest route.

    """
    if current_best_length is None:
        # Assigning value neutral for further calculations.
        current_best_length = 1

    population = [
        RudolphRoute(mutation_proba=mutation_proba)
        for _ in range(precursors)
    ]

    for i in range(generations):
        chosen, best = _select_descendants(
            population, current_best_length, chosen_size
        )
        if i == 0:
            alpha_chrom = best
            current_best_length = alpha_chrom.length()
            if verbose:
                print(
                    "Benchmark route's length taken from the precursors' "
                    "population: {0:.2f}. It passes through the following "
                    "cities: {1}. Mutation probability at the beginning was: "
                    "{2:.2f}%."
                    .format(
                        current_best_length,
                        alpha_chrom.route,
                        alpha_chrom.mutation_proba * 100
                    )
                )
        elif best.length() < alpha_chrom.length():
            alpha_chrom = best
            current_best_length = alpha_chrom.length()
            if verbose:
                print(
                    "After {0} generations the best route's length is: {1:.2f}"
                    " and it passes through the following cities: {2}."
                    " Probability of mutation is currently at: {3:.2f}%."
                    .format(
                        i + 1,
                        current_best_length,
                        alpha_chrom.route,
                        alpha_chrom.mutation_proba * 100
                    )
                )
        population = [
            _dim_mutation(rudolph, mutation_dimming_factor)
            for rudolph in _generate_population(chosen)
        ]

    _, final_leader = _pick_best(population, current_best_length)

    if final_leader.length() < alpha_chrom.length():
        alpha_chrom = final_leader

    return alpha_chrom


def _select_descendants(population, current_best_length, chosen_size=.05):
    """Selects best parents - mating pool.

    Assigns the greatest probability of choosing to chromosomes with the
    shortest routes and then selects randomly descendants taking weights into
    account.

    Parameters
    ----------
    population: list
        Collection of RudolphRoutes

    current_best_length: float
        Length of the best route found so far.

    chosen_size: int, float
        Number of chromosomes that will be selected if "int" is passed or
        fraction of the whole population that will be selected for float.

    Returns
    -------
    chosen: list
        Collection of the chosen chromosomes.

    best: RudolphRoute
        The fittest object from the population.

    """
    population_fitness, best = _pick_best(population, current_best_length)

    total_fitness = sum(population_fitness)
    choosing_proba = [
        fitness / total_fitness for fitness in population_fitness
    ]

    if isinstance(chosen_size, float):
        assert 0 <= chosen_size <= 1, (
            'Fraction of the chosen subpopulation cannot exceed 1 and be '
            'lower than 0!'
        )
        chosen_size = int(len(population) * chosen_size)

    chosen = np.random.choice(
        population, size=chosen_size, replace=False, p=choosing_proba
    )

    return chosen, best


def _generate_population(parents):
    """Creates new population based on the given parents.

    Parameters
    ----------
    parents: list
        Collection of parents for which childrens will be created.

    Returns
    -------
    population: list
        Collection of childs.
    """
    population = []
    for idx, chrom in enumerate(parents[:-1]):
        for other_chrom in parents[idx + 1:]:
            new_chrom = chrom + other_chrom
            population.append(new_chrom)

    return population


def _pick_best(population, current_best_length):
    """Selects best chromosome from the population and its fitness."""
    population_fitness = [
        1 / (chrom.length() / current_best_length) for chrom in population
    ]
    best_idx = population_fitness.index(max(population_fitness))
    best = population[best_idx]

    return population_fitness, best


def _dim_mutation(rudolph_object, mutation_dimming_factor):
    """Dims mutation_proba attribute of the RudolphRoute object."""
    rudolph_object.mutation_proba *= mutation_dimming_factor
    return rudolph_object
