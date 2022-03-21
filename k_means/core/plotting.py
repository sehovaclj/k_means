"""Module to plot the simulation and iteration results of the algorithm."""
from typing import List
import matplotlib.pyplot as plt
from k_means.utils.plotting import scatter_plot_initial_dists, \
    scatter_plot_initial_dists_no_colour, plot_initial_centroids, \
    plot_new_centroids, clear_old_clusters_and_plot_new_ones
from k_means.utils.mapping import Parameters
from k_means.core.data_prep import DataEng


def plot_simulation(parameters: Parameters,
                    data_eng: DataEng,
                    results: List[any]) -> None:
    """Structure of plotting the k-means algorithm results.

    Args:
        parameters: see k_means.utils.mapping.Parameters for more details.
        data_eng: see k_means.core.data_prep.DataEng for more details.
        results: list containing results of the k-means algorithm run.

    Returns:
        None.
    """
    print('Plotting initial distributions and simulating the K-means clustering results')
    # plot the initial distributions
    scatter_plot_initial_dists(parameters, data_eng)
    # plot distributions without colour. Keep track of this figure for remaining K-means clustering
    plt.figure(2)
    plots = scatter_plot_initial_dists_no_colour(parameters, data_eng)
    plt.pause(parameters.pause_length)
    # add centroids to colourless plot
    plot_cens = plot_initial_centroids(data_eng)
    plt.pause(parameters.pause_length)
    # now plot the results of the simulation
    for results_iter in results:
        if results_iter['iteration'] > 0:
            plt.legend().remove()
        # clear old clusters from plot (figure 2)
        plots = clear_old_clusters_and_plot_new_ones(parameters, plots, results_iter)
        plt.legend()
        plt.pause(parameters.pause_length)
        # remove old centroids and legend
        plot_cens.remove()
        # plot new centroids
        plot_cens = plot_new_centroids(results_iter)
        plt.legend()
        plt.pause(parameters.pause_length)
    # end of results loop
    print('End of program, plots will be active for 20 seconds')
    plt.pause(20.0)
