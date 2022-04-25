"""Module with functions that run the main k-means algorithm."""
from typing import List

from k_means.core.data_prep import DataEng
from k_means.utils.algorithm import calculate_distances, find_new_centroids, append_to_results, \
    check_for_convergence
from k_means.utils.mapping import Parameters


def k_means_algorithm(parameters: Parameters,
                      data_eng: DataEng) -> List[any]:
    """Summarizes the main steps of the algorithm.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.

    Returns:
        results: list containing results of the k-means algorithm.
    """
    # store results for plotting
    results = []
    counter = 0
    convergence = False
    print('Starting K-Means Algorithm...')
    while not convergence:
        data_eng = calculate_distances(parameters, data_eng)
        data_eng = find_new_centroids(parameters, data_eng)
        results.append(append_to_results(parameters, data_eng, counter))
        data_eng, convergence = check_for_convergence(parameters, data_eng, counter)
        counter += 1
    return results
