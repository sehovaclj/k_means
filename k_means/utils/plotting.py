"""Module to contain functions used in core algorithm."""
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from typing import List, Dict


def scatter_plot_initial_dists(num_dists: int,
                               add_noise: bool,
                               x_dists: list,
                               y_dists: list) -> None:
    plt.figure(1)
    # add points to scatter plot
    for i in range(num_dists):
        if add_noise:
            if i != num_dists - 1:
                plt.scatter(x_dists[i], y_dists[i], label='Dist ' + str(i + 1))
            elif i == num_dists - 1:
                plt.scatter(x_dists[i], y_dists[i], label='Noise Dist')
        elif not add_noise:
            plt.scatter(x_dists[i], y_dists[i], label='Dist ' + str(i + 1))
    plt.title('Initial distributions/samples in colour')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()


def scatter_plot_initial_dists_no_colour(num_dists: int,
                                         x_dists: list,
                                         y_dists: list) -> List[any]:
    plt.figure(2)
    plots = []
    for i in range(num_dists):
        plots.append({'plot_' + str(i + 1): plt.scatter(x_dists[i], y_dists[i], c='k')})
    plt.title('K-means Clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    return plots


def plot_initial_dists(num_dists: int,
                       add_noise: bool,
                       x_dists: list,
                       y_dists: list,
                       pause_length: float) -> List[any]:
    # create the initial scatter plot
    scatter_plot_initial_dists(num_dists, add_noise, x_dists, y_dists)
    # Now plot the above, but without colour. We will keep track of this figure for remaining K-means clustering
    plots = scatter_plot_initial_dists_no_colour(num_dists, x_dists, y_dists)
    plt.pause(pause_length)
    return plots


def add_centroids_to_plots(centroids_prev: np.array,
                           boundaries: Dict[str, float],
                           pause_length: float) -> [any, any]:
    plot_centroids = plt.scatter(centroids_prev[:, 0], centroids_prev[:, 1],
                                 c='r', marker='*', s=200, label='Initial Centroids')
    legend_centroids = plt.legend()
    plt.xlim(left=boundaries['min_x'] - 1.0, right=boundaries['max_x'] + 1.0)
    plt.ylim(bottom=boundaries['min_y'] - 1.0, top=boundaries['max_y'] + 1.0)
    plt.pause(pause_length)
    return plot_centroids, legend_centroids


def remove_plots(num_dists: int,
                 plots: List[any]) -> List[any]:
    for i in range(num_dists):
        plots[i]['plot_' + str(i + 1)].remove()
    return plots


def choose_colours(num_clusters: int) -> np.array:
    colours = np.zeros([num_clusters, 1, 3])
    for i in range(num_clusters):
        colours[i] = [random.random(), random.random(), random.random()]
    return colours
