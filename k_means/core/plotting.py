"""Module to plot the simulation and iteration results of the algorithm."""
import matplotlib.pyplot as plt
from typing import List, Dict
from k_means.utils.plotting import scatter_plot_initial_dists, \
    scatter_plot_initial_dists_no_colour, plot_initial_centroids, \
    append_to_scatter_plot, plot_new_centroids


def plot_simulation(parameters,
                    data_eng,
                    results):
    # plot the initial distributions
    scatter_plot_initial_dists(parameters, data_eng)
    # plot distributions without colour. Keep track of this figure for remaining K-means clustering
    plt.figure(2)
    plots = scatter_plot_initial_dists_no_colour(parameters, data_eng)
    plt.pause(parameters.pause_length)
    # add centroids to colourless plot
    plot_cens, leg_cens = plot_initial_centroids(data_eng)
    plt.pause(parameters.pause_length)
    # now plot the results of the simulation
    for results_iter in results:
        # plot the clusters
        plots = []
        for i in range(parameters.num_clusters):
            plots.append(append_to_scatter_plot(results_iter, i))
        leg_clusters = plt.legend()
        plt.pause(parameters.pause_length)
        # remove old centroids and legend
        plot_cens.remove()
        leg_cens.remove()
        # plot new centroids
        plot_cens = plot_new_centroids(results_iter)
        leg_cens = plt.legend()
        plt.pause(parameters.pause_length)
        # clear old clusters from plot (figure 2)
        for i in range(parameters.num_clusters):
            plots[i]['plot_' + str(i + 1)].remove()
        leg_clusters.remove()
    # end of results loop
    plt.show()
