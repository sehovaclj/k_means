"""Module to help organize main algorithm."""
from k_means.utils.algorithm import create_random_dists, dists_as_matrix, \
    dists_min_max, initial_centroids


def main_data_engineering(num_clusters: int,
                          num_dists: int,
                          num_samples: int,
                          add_noise: bool):
    # create the random distributions
    x_dists, y_dists, num_dists = create_random_dists(num_dists, num_samples, add_noise)
    # A now contains all distributions in (x,y) coordinate form
    matrix_dists = dists_as_matrix(num_dists, x_dists, y_dists)
    # find min/max values of the x and y distributions. Will be used as boundaries for the initial random centroids
    boundaries = dists_min_max(x_dists, y_dists)
    # obtain initial random (uniform) centroids
    centroids_prev = initial_centroids(num_clusters, boundaries)
    return x_dists, y_dists, num_dists, matrix_dists, boundaries, centroids_prev
