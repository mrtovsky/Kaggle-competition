import numpy as np

import prepare_example_data

from paralytics.mathy import check_prime
from paralytics.utils.importing import get_csv
from random import randint, sample, uniform


class RudolphRoute(object):
    """Chromosome for a genetic algorithm.

    Parameters
    ----------
    cities: set
        Set of city coordinates that are avaliable for creating a route.

    route: list
        Route of subsequent cities containing all of the "cities" elements
        exactly once.

    mutation_proba: float
        Probability of swapping two cities on the route in the mutation
        process.

    """
    for _ in range(2):
        try:
            _data = get_csv('cities.csv')
        except FileNotFoundError as e:
            e.args += ('Generating random cities.csv data instead.',)
            print('. '.join(e.args[1:]))
            prepare_example_data.create_csv(csv_name='cities')
            continue
        break

    def __init__(self, cities=None, route=None, mutation_proba=.01):
        self.cities = cities
        self.route = route
        self.mutation_proba = mutation_proba

    @property
    def cities(self):
        return self._cities

    @property
    def route(self):
        return self._route

    @cities.setter
    def cities(self, s):
        """When no cities are given take all of them except the North Pole."""

        if s is None:
            s = set(range(1, len(self._data)))
        assert isinstance(s, set), \
            'The cities must be an instance of a set!'
        self._cities = s

    @route.setter
    def route(self, arr):
        """When no route is specified generate a random route."""

        if arr is None:
            arr = sample(self.cities, len(self.cities))
        assert isinstance(arr, list), \
            'The route of subsequent cities must be passed as a list!'
        assert len(set(arr)) == len(arr), \
            'The route must contain every city exactly once!'
        assert set(arr) == self.cities, \
            "The route must be the cities' subset and contain all of them!"
        self._route = arr

    def length(self, route=None):
        """Measure the total route distance."""

        fitness = 0

        if route is None:
            route = self.route

        road = route.copy()
        # Adding a starting and destination point to the route.
        road.insert(0, 0)
        road.insert(len(road), 0)

        for idx in range(len(road) - 1):
            distance = self._euclidean_norm(
                self._data[road[idx]], self._data[road[idx + 1]]
            )

            if not (idx + 1) % 10 and not check_prime(road[idx]):
                distance = distance * 1.1

            fitness += distance

        return fitness

    def __add__(self, obj):
        """Selects the best child of the parents via crossover and mutation."""

        routes = self._crossover(obj)
        mutated_routes = [self._mutate(route) for route in routes]
        reversed_routes = [route[::-1] for route in mutated_routes]
        mutated_routes.extend(reversed_routes)
        measured_routes = [
            (route, self.length(route)) for route in mutated_routes
        ]
        best_route = min(measured_routes, key=lambda pair: pair[1])[0]

        new_obj = type(self)(
            cities=self.cities,
            route=best_route,
            mutation_proba=self.mutation_proba
        )
        return new_obj

    def __radd__(self, obj):
        return self + obj

    def __reversed__(self):
        rev_route = self.route[::-1]
        return type(self)(cities=self.cities, route=rev_route)

    def _crossover(self, obj):
        """Checking conditions and determining the list of children."""
        try:
            route = obj.route
            assert self.cities == obj.cities, (
                'Adding is only possible for routes defined on '
                'the same cities!'
            )
        except AttributeError:
            assert \
                len(self.route) == len(obj) and set(self.route) == set(obj), (
                    'Adding is only possible for routes defined on the same'
                    'cities and being the same length!'
                )
            route = obj

        start = randint(0, len(self.route) - 1)
        stop = randint(start + 1, len(self.route))

        cuts = [set(self.route[start:stop]), set(route[start:stop])]
        routes = [self.route, route]
        new_routes = []

        for idx in range(2):
            # This trick in brackets always transforms 0 to 1 and 1 to 0.
            other_route = routes[abs(idx - 1)]

            new_route = [
                city for city in routes[idx] if city not in cuts[abs(idx - 1)]
            ]
            new_route[start:start] = other_route[start:stop]
            new_routes.append(new_route)

        return new_routes

    def _mutate(self, route):
        """Mutating route."""
        road = route.copy()

        for idx, city in enumerate(road):
            if uniform(0, 1) > self.mutation_proba:
                continue
            else:
                swap_idx = randint(0, len(road) - 1)
                road[idx], road[swap_idx] = road[swap_idx], road[idx]

        return road

    @staticmethod
    def _euclidean_norm(v, w):
        return np.sqrt(abs(v['X'] - w['X']) ** 2 + abs(v['Y'] - w['Y']) ** 2)
