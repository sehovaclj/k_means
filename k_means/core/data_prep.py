"""Module to help organize main algorithm."""
from k_means.utils.mapping import Parameters
from k_means.utils.algorithm import create_random_dists, dists_as_matrix, \
    dists_min_max, initial_centroids, clusters_list
from k_means.utils.plotting import choose_colours


class DataEng:
    """Contains data engineering parameters.

    """

    def __init__(self,
                 x_dists,
                 y_dists,
                 matrix_dists,
                 boundaries,
                 centroids_prev,
                 clusters,
                 colours):
        self.x_dists = x_dists
        self.y_dists = y_dists
        self.matrix_dists = matrix_dists
        self.boundaries = boundaries
        self.centroids_prev = centroids_prev
        self.centroids_new = centroids_prev
        self.initial_centroids = centroids_prev
        self.clusters = clusters
        self.colours = colours


def main_data_engineering(parameters: Parameters) -> [DataEng, Parameters]:
    # create the random distributions
    x_dists, y_dists, parameters = create_random_dists(parameters)
    # A now contains all distributions in (x,y) coordinate form
    matrix_dists = dists_as_matrix(parameters.num_dists, x_dists, y_dists)
    # find min/max values of the x and y distributions. Will be used as boundaries for the initial random centroids
    boundaries = dists_min_max(x_dists, y_dists)
    # obtain initial random (uniform) centroids
    centroids_prev = initial_centroids(parameters.num_clusters, boundaries)
    # create clusters list of lists
    clusters = clusters_list(parameters.num_clusters)
    # choose colours to use during plotting
    colours = choose_colours(parameters.num_clusters)
    return DataEng(x_dists,
                   y_dists,
                   matrix_dists,
                   boundaries,
                   centroids_prev,
                   clusters,
                   colours), parameters
