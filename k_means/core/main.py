"""Main sequence of functions to run algorithm."""
from typing import Dict
import numpy as np
from k_means.utils.mapping import Parameters
from k_means.core.data_prep import main_data_engineering
from k_means.core.algorithm import k_means_algorithm
from k_means.core.plotting import plot_simulation


def run(message: Dict[str, any]) -> None:
    """Main sequence of functions needed to run the algorithm
        and plot the results as a simulation. Helps keep the program organized.

    Args:
        message: see k_means.resources.input

    Returns:
        None.
    """
    print('\nStarting program')
    # get parameters from post request message
    parameters = Parameters(message)
    # preserving random state
    np.random.seed(parameters.seed)
    # main data engineering is first
    data_eng, parameters = main_data_engineering(parameters)
    # main k-means algorithm, return results
    results = k_means_algorithm(parameters, data_eng)
    # plot initial distributions and results of our k means algorithm
    plot_simulation(parameters, data_eng, results)
