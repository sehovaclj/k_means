"""Module to run the main k-means algorithm."""
import numpy as np
from math import sqrt
from k_means.utils.mapping import Parameters
from k_means.core.data_prep import DataEng
from k_means.utils.algorithm import clusters_list


def k_means_algorithm(parameters: Parameters,
                      data_eng: DataEng):
    # store results for plotting
    results = []
    counter = 0
    convergence = False
    while not convergence:
        data_eng = calculate_distances(parameters, data_eng)
        data_eng = find_new_centroids(parameters, data_eng)
        results.append(append_to_results(parameters, data_eng, counter))
        data_eng, convergence = test_convergence(parameters, data_eng, counter)
        counter += 1
    return results


def calculate_distances(parameters: Parameters,
                        data_eng: DataEng):
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


def find_new_centroids(parameters,
                       data_eng):
    # compute mean of cluster -- and make this our new centroid for that cluster
    data_eng.centroids_new = data_eng.centroids_prev.copy()
    for i in range(parameters.num_clusters):
        data_eng.centroids_new[i] = [np.array(data_eng.clusters[i])[:, 0].mean(),
                                     np.array(data_eng.clusters[i])[:, 1].mean()]
    return data_eng


def append_to_results(parameters,
                      data_eng,
                      counter):
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


def test_convergence(parameters,
                     data_eng,
                     counter):
    """Test for convergence.
    Easiest way, also saves most memory, is to test if the centroids distance from the
        previous centroid is less than epsilon (usually a small value).
        Hence, if the centroid has not moved or barely moved, the desired number of clusters has been obtained

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
