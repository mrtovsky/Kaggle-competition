import numpy as np

from chromosome import RudolphRoute


def start_evolution(
    current_best_length=None, precursors=200,
    chosen_size=20, generations=500, verbose=1
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

    precursors = [RudolphRoute() for _ in range(precursors)]

    population = precursors
    for i in range(generations):
        chosen, best = _select_descendants(
            population, current_best_length, chosen_size
        )
        if i == 0:
            alpha_chrom = best
            current_best_length = alpha_chrom.length()
        elif best.length() < alpha_chrom.length():
            alpha_chrom = best
            current_best_length = alpha_chrom.length()
            if verbose:
                print(
                    "After {0} generations the best route's length is: {1:.2f}"
                    " and it passes through the following cities: {2}."
                    .format(i, alpha_chrom.length(), alpha_chrom.route)
                )
        population = _generate_population(chosen)

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
