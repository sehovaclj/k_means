"""Module that contains functions called during the core algorithm run."""
import numpy as np
import numpy.random as random
from typing import Dict
from k_means.utils.mapping import Parameters


def create_random_dists(parameters: Parameters) -> [list, list, int]:
    print('Creating random initial distributions')
    # empty lists to store dists
    x_dists, y_dists = [], []
    # creating random distributions, with a slight random shift
    for i in range(parameters.num_dists):
        x_dists.append(random.randn(parameters.num_samples) +
                       (random.randint(2, 6) * random.randn()))
        y_dists.append(random.randn(parameters.num_samples) +
                       (random.randint(2, 6) * random.randn()))
    # if a larger noise distribution is desired, set add_noise=True
    if parameters.add_noise:
        x_dists.append(random.randn(parameters.num_samples) *
                       (random.randint(4, 8) * random.randn()))
        y_dists.append(random.randn(parameters.num_samples) *
                       (random.randint(4, 8) * random.randn()))
        # we've added a noise dist, so increase the num dists by 1
        parameters.num_dists = parameters.num_dists + 1
    return x_dists, y_dists, parameters


def dists_as_matrix(num_dists: int,
                    x_dists: list,
                    y_dists: list) -> np.array:
    # convert distribution lists to Matrices
    matrix_dists = []
    for i in range(num_dists):
        matrix_dists.append(np.array([x_dists[i], y_dists[i]]).transpose())
    # shape here == num_dists x num_samples x 2 (the 2 here represents the (x,y) coordinate pairing)
    matrix_dists = np.array(matrix_dists)
    return matrix_dists


def dists_min_max(x_dists: list,
                  y_dists: list) -> Dict[str, float]:
    boundaries = {
        'min_x': np.array(x_dists).min(),
        'min_y': np.array(y_dists).min(),
        'max_x': np.array(x_dists).max(),
        'max_y': np.array(y_dists).max()
    }
    return boundaries


def initial_centroids(num_clusters: int,
                      boundaries: Dict[str, float]) -> np.array:
    print('Choosing initial centroids randomly')
    centroids_prev = []
    for i in range(num_clusters):
        centroids_prev.append([random.uniform(boundaries['min_x'], boundaries['max_x']),
                               random.uniform(boundaries['min_y'], boundaries['max_y'])])
    return np.array(centroids_prev)


def clusters_list(num_clusters):
    clusters = []
    for i in range(num_clusters):
        clusters.append([])
    return clusters
