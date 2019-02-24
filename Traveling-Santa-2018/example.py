import numpy as np
import random

import genetic_algorithm


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    genetic_algorithm.start_evolution(generations=100, verbose=1)


if __name__ == '__main__':
    main()
