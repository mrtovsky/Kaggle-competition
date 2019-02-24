import pandas as pd
import random


def create_csv(csv_name, number_of_cities=10, start=0, stop=100):
    """Prepares example data for Rudolph to travel to."""
    df = pd.DataFrame({
        'CityId': list(range(number_of_cities)),
        'X': [random.uniform(start, stop) for _ in range(number_of_cities)],
        'Y': [random.uniform(start, stop) for _ in range(number_of_cities)]
    })

    df.to_csv('.'.join((csv_name, 'csv')), index=False)
