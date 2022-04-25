"""Module to contain functions used in core algorithm."""
from typing import List, Dict

import matplotlib.pyplot as plt

from k_means.core.data_prep import DataEng
from k_means.utils.mapping import Parameters


def scatter_plot_initial_dists(parameters: Parameters,
                               data_eng: DataEng) -> None:
    """Plot the initial distributions in colour, to identify them before the algorithm and simulation.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.

    Returns:
        None.
    """
    plt.figure(1)
    # add points to scatter plot
    for i in range(parameters.num_dists):
        if parameters.add_noise:
            if i != parameters.num_dists - 1:
                plt.scatter(data_eng.x_dists[i], data_eng.y_dists[i], label='Dist ' + str(i + 1))
            elif i == parameters.num_dists - 1:
                plt.scatter(data_eng.x_dists[i], data_eng.y_dists[i], label='Noise Dist')
        elif not parameters.add_noise:
            plt.scatter(data_eng.x_dists[i], data_eng.y_dists[i], label='Dist ' + str(i + 1))
    plt.title('Initial distributions/samples in colour')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()


def scatter_plot_initial_dists_no_colour(parameters: Parameters,
                                         data_eng: DataEng) -> List[any]:
    """Plot the data points with no colour, to visualize the points prior to algorithm.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.

    Returns:
        plots: list of dicts containing the scatter plots.
    """
    plots = []
    for i in range(parameters.num_dists):
        plots.append({'plot_' + str(i + 1): plt.scatter(
            data_eng.x_dists[i],
            data_eng.y_dists[i],
            c='k')})
    plt.title('K-means Clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    return plots


def plot_initial_centroids(data_eng: DataEng) -> plt.scatter:
    """Plot the initial centroids and give the plot limits to visualize better.

    Args:
        data_eng: see k_means.core.data_prep.DataEng for more details.

    Returns:
        plot_cens: scatter plot of the centroids.
    """
    plot_cens = plt.scatter(data_eng.initial_centroids[:, 0],
                            data_eng.initial_centroids[:, 1],
                            c='r', marker='*', s=200, label='Initial Centroids')
    plt.legend()
    plt.xlim(left=data_eng.boundaries['min_x'] - 1.0,
             right=data_eng.boundaries['max_x'] + 1.0)
    plt.ylim(bottom=data_eng.boundaries['min_y'] - 1.0,
             top=data_eng.boundaries['max_y'] + 1.0)
    return plot_cens


def append_to_scatter_plot(results_iter: Dict[any, any],
                           i: int) -> Dict[any, any]:
    """Return a dict that will append a scatter plot to a list. Mainly used to keep code organized and neat.

    Args:
        results_iter: dict containing results of that iteration of the simulation.
        i: cluster index.

    Returns: dict containing scatter plots.
    """
    return {'plot_' + str(i + 1): plt.scatter(
        results_iter['cluster_plots'][i]['plot_x_' + str(i + 1)],
        results_iter['cluster_plots'][i]['plot_y_' + str(i + 1)],
        c=results_iter['cluster_plots'][i]['plot_c_' + str(i + 1)],
        label=results_iter['cluster_plots'][i]['plot_label_' + str(i + 1)])}


def plot_new_centroids(results_iter: Dict[any, any]) -> plt.scatter:
    """Plots the new centroids for that iteration.

    Args:
        results_iter: dict containing results information of that iteration of the simulation.

    Returns:
        scatter plot containing new centroids.
    """
    plot_cens = plt.scatter(
        results_iter['new_centroids']['x'],
        results_iter['new_centroids']['y'],
        c=results_iter['new_centroids']['c'],
        marker=results_iter['new_centroids']['marker'],
        s=results_iter['new_centroids']['s'],
        label=results_iter['new_centroids']['label'])
    return plot_cens


def clear_old_clusters_and_plot_new_ones(parameters: Parameters,
                                         plots: List[any],
                                         results_iter: Dict[any, any]) -> List[any]:
    """Remove old clusters from plots and append new ones. This is needed for the visualization process.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        plots: list of dicts containing scatter plots.
        results_iter: results containing information of the iteration of the simulation.

    Returns:
        plots: list of dicts containing scatter plots.
    """
    # clear old clusters from plot (figure 2)
    for i in range(parameters.num_clusters):
        plots[i]['plot_' + str(i + 1)].remove()
    # plot the clusters
    plots = []
    for i in range(parameters.num_clusters):
        plots.append(append_to_scatter_plot(results_iter, i))
    return plots
