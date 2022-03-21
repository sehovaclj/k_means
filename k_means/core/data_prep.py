"""Module that contains main data prep."""
from k_means.utils.mapping import Parameters
from k_means.utils.algorithm import create_random_dists, dists_as_matrix, \
    dists_min_max, initial_centroids, clusters_list, choose_colours
from typing import Tuple


class DataEng:
    """Contains data engineering parameters.
        Main purpose: Helps keep code organized.

    Args:
        x_dists: random numbers in respective distribution, in the x-axis.
        y_dists: random numbers in respective distribution, in the y-axis.
        matrix_dists: (x, y) points from above, hence the x and y dists in matrix form.
        boundaries: limits in x and y direction.
        centroids_prev: the previous centroids, or the centroids on the previous iteration.
        clusters: the data points need to be assigned to certain clusters.
        colours: this is for plotting, just N random colours.

    Attributes:
        centroids_new: when we find new centroids, need to assign them a new variable.
        initial_centroids: very first centroids assigned, these will be randomly generated.

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


def main_data_engineering(parameters: Parameters) -> Tuple[DataEng, Parameters]:
    """Main structure of initial data engineering and prep to generate the
        initial distributions, centroids, and colours used to identify clusters.
        This function basically sets up the problem for us to be able to run our k-means algorithm.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.

    Returns:
        DataEng: see k_means.core.data_prep.DataEng for more details.
        parameters: if we alter the parameters, we need to return the altered version.
    """
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
