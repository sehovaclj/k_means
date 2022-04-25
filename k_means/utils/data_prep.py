"""Module that contains functions called during the core algorithm run."""
import sys
from inspect import currentframe
from typing import Dict, List

import numpy as np
from numpy import random

from k_means.utils.exception_log_manager import print_detailed_exception
from k_means.utils.mapping import Parameters


def create_random_dists(parameters: Parameters) -> [list, list, int]:
    """Creates the initial distributions.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.

    Returns:
        x_dists: random numbers of the distributions in the x-axis.
        y_dists: random numbers of the distributions in the y-axis.
        parameters: if we alter any of the parameters, need to return an updatded verion.
    """
    print('Creating random initial distributions')
    # empty lists to store dists
    x_dists, y_dists = [], []
    # creating random distributions, with a slight random shift
    for _ in range(parameters.num_dists):
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
    """Convert distribution parameters into an array for ease of use.

    Args:
        num_dists: number of distributions used.
        x_dists: random numbers of the distributions in the x-axis.
        y_dists: random numbers of the distributions in the y-axis.

    Returns:
        matrix_dists: x and y distributions as points in a matrix.
    """
    # convert distribution lists to Matrices
    matrix_dists = []
    try:
        for i in range(num_dists):
            matrix_dists.append(np.array([x_dists[i], y_dists[i]]).transpose())
    except IndexError as index_expt:
        print_detailed_exception(currentframe().f_code.co_name, sys.exc_info(), index_expt)
    # shape here == num_dists x num_samples x 2 (the 2 here represents the (x,y) coordinate pairing)
    matrix_dists = np.array(matrix_dists)
    return matrix_dists


def dists_min_max(x_dists: list,
                  y_dists: list) -> Dict[str, float]:
    """Find limits of x and y distributions.

    Args:
        x_dists: random numbers of the distributions in the x-axis.
        y_dists: random numbers of the distributions in the y-axis.

    Returns:
        boundaries: min and max in both the x and y direction.
    """
    try:
        boundaries = {
            'min_x': np.array(x_dists).min(),
            'min_y': np.array(y_dists).min(),
            'max_x': np.array(x_dists).max(),
            'max_y': np.array(y_dists).max()
        }
    except ValueError as value_expt:
        print_detailed_exception(currentframe().f_code.co_name, sys.exc_info(), value_expt)
        boundaries = {}
    return boundaries


def initial_centroids(num_clusters: int,
                      boundaries: Dict[str, float]) -> np.array:
    """Create random initial centroids within our limits.

    Args:
        num_clusters: number of clusters to find, hence number of needed initial centroids.
        boundaries: need the limits in which to randomly choose initial centroids.

    Returns:
        np.array(centroids_prev): array containing initial centroids.
    """
    print('Choosing initial centroids randomly')
    centroids_prev = []
    try:
        for _ in range(num_clusters):
            centroids_prev.append([random.uniform(boundaries['min_x'], boundaries['max_x']),
                                   random.uniform(boundaries['min_y'], boundaries['max_y'])])
    except KeyError as key_expt:
        print_detailed_exception(currentframe().f_code.co_name, sys.exc_info(), key_expt)
    return np.array(centroids_prev)


def clusters_list(num_clusters: int) -> List[list]:
    """Create a list of empty lists for number of desired clusters.

    Args:
        num_clusters: number of clusters to find, we will be storing points in these empty indexed lists.

    Returns:
        clusters: empty list of lists.
    """
    clusters = []
    for _ in range(num_clusters):
        clusters.append([])
    return clusters


def choose_colours(num_clusters: int) -> np.array:
    """"Randomly choose colours equivalent to the number of clusters we need to find.

    Args:
        num_clusters: desired number of clusters to find.

    Returns:
        colours: array of randomly chosen numbers which == a colour when plotting.
    """
    colours = np.zeros([num_clusters, 1, 3])
    for i in range(num_clusters):
        colours[i] = [random.random(), random.random(), random.random()]
    return colours
