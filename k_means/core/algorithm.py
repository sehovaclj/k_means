"""Module with functions that run the main k-means algorithm."""
import numpy as np
from math import sqrt
from k_means.utils.mapping import Parameters
from k_means.core.data_prep import DataEng
from k_means.utils.algorithm import clusters_list
from typing import List, Dict


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
        data_eng, convergence = test_convergence(parameters, data_eng, counter)
        counter += 1
    return results


def calculate_distances(parameters: Parameters,
                        data_eng: DataEng) -> DataEng:
    """For each sample in each distribution, calculate the distance
        to each cluster (Euclidean distance), then find the minimum distance cluster index
        and add that point to that cluster.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.

    Returns:
        data_eng: since we change the clusters,
            we need to update the data_eng.cluster attribute and return data_eng.
    """
    # for each sample in each distribution...
    for dist in range(parameters.num_dists):
        for sample in range(parameters.num_samples):
            distances = []
            # for each sample, compute distance to each centroid
            for cluster in range(parameters.num_clusters):
                distances.append(
                    sqrt((data_eng.matrix_dists[dist][sample][0] - data_eng.centroids_prev[cluster][0]) ** 2 + (
                            data_eng.matrix_dists[dist][sample][1] - data_eng.centroids_prev[cluster][1]) ** 2))
            # find smallest distance
            c_idx = np.array(distances).argmin()
            # append sample, or point, to that cluster
            data_eng.clusters[c_idx].append(data_eng.matrix_dists[dist][sample])
    return data_eng


def find_new_centroids(parameters: Parameters,
                       data_eng: DataEng) -> DataEng:
    """Find new cluster centroids by taking the mean
        of the data points in the cluster,
        Take the mean of all x points: this is our x of the new centroid,
        Take the mean of all y points: this is our y of the new centroid.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.

    Returns:
        data_eng: since we change the centroids,
            we need to update the data_eng.centroids_new attribute and return data_eng.
    """
    data_eng.centroids_new = data_eng.centroids_prev.copy()
    for i in range(parameters.num_clusters):
        data_eng.centroids_new[i] = [np.array(data_eng.clusters[i])[:, 0].mean(),
                                     np.array(data_eng.clusters[i])[:, 1].mean()]
    return data_eng


def append_to_results(parameters: Parameters,
                      data_eng: DataEng,
                      counter: int) -> Dict[any, any]:
    """Create a dict to store the results for plotting.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.
        counter: the iteration number.

    Returns:
        results_iter: dict of results used in plotting the simulation.
    """
    results_iter = {
        'iteration': counter,
        'cluster_plots': [],
        'new_centroids': {}
    }
    for i in range(parameters.num_clusters):
        results_iter['cluster_plots'].append({
            'plot_x_' + str(i + 1): np.array(data_eng.clusters[i])[:, 0],
            'plot_y_' + str(i + 1): np.array(data_eng.clusters[i])[:, 1],
            'plot_c_' + str(i + 1): data_eng.colours[i],
            'plot_label_' + str(i + 1): 'Cluster ' + str(i + 1)
        })
    # add new centroids to results
    results_iter['new_centroids']['x'] = data_eng.centroids_new[:, 0]
    results_iter['new_centroids']['y'] = data_eng.centroids_new[:, 1]
    results_iter['new_centroids']['c'] = 'r'
    results_iter['new_centroids']['marker'] = '*'
    results_iter['new_centroids']['s'] = 200
    results_iter['new_centroids']['label'] = 'Centroids Iter ' + str(counter + 1)
    return results_iter


def test_convergence(parameters: Parameters,
                     data_eng: DataEng,
                     counter: int) -> [DataEng, bool]:
    """Test for convergence.
        Easiest way, also saves most memory, is to test if the centroids distance from the
        previous centroid is less than epsilon (usually a small value).
        Hence, if the centroid has not moved or barely moved, the desired number of clusters has been obtained

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.
        counter: the iteration number.

    Returns:
        data_eng: if we have not reached convergence,
            we need to reset our clusters and assign our new found centroids as the previous ones.
        convergence: True or False, have we reached convergence or not.
    """
    distances_bool = []
    for i in range(parameters.num_clusters):
        dist_cen = sqrt(
            (data_eng.centroids_new[i][0] - data_eng.centroids_prev[i][0]) ** 2 + (
                    data_eng.centroids_new[i][1] - data_eng.centroids_prev[i][1]) ** 2)
        if dist_cen <= parameters.eps:
            distances_bool.append(True)
        elif dist_cen > parameters.eps:
            distances_bool.append(False)
    # if all distances are under a certain eps threshold, end algorithm;
    # if not, assign new centroids as previous, and clear clusters from plots
    if all(distances_bool):
        print('\nConvergence achieved in {} iterations'.format(counter + 1))
        convergence = True
    elif not all(distances_bool):
        data_eng.centroids_prev = data_eng.centroids_new
        data_eng.clusters = clusters_list(parameters.num_clusters)
        convergence = False
    # also test if max number of iterations has been reached
    if counter == parameters.max_iter:
        print('Convergence timed out by Maximum number of iterations')
        convergence = True
    return data_eng, convergence
